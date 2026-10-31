"""
VISENTRA <-> HiVT adapter (EXPERIMENTAL image-plane trajectory forecasting).

Two parts, deliberately separated so the main VISENTRA process never needs PyTorch:

  * HiVTTrackHistory  - numpy only. Lives in the main process. Keeps one rolling list of REAL
                        timestamped observations per ByteTrack ID.
  * HiVTPredictor     - needs torch + the HiVT repo. Lives in the dedicated HiVT worker process.

What HiVT actually expects (verified against HiVT/datasets/argoverse_v1_dataset.py and
models/hivt.py, models/decoder.py):

  * 20 history steps + 30 future steps at 10 Hz (dt = 0.1 s), positions in METRES.
  * x[N,20,2]        per-step DISPLACEMENTS (x[:,0] = 0, x[:,t] = pos[t] - pos[t-1], 0 if either missing)
  * positions[N,50,2] absolute positions in a shared scene frame (future part unused at inference)
  * padding_mask[N,50] True = unobserved. Future steps (20..49) are always True at inference.
  * bos_mask[N,20]   True where an actor's track begins
  * rotate_angles[N] each actor's heading (atan2 of its last two valid history positions)
  * edge_index       fully connected actor graph (permutations), distance-dropped inside the model
  * lane_* tensors   Argoverse HD-map lane features. A monocular camera has none, so they are EMPTY
                     (structurally valid; the model simply gets no map context - see limitations).
  * output y_hat[F=6, N, 30, 4] = (loc x, loc y, scale x, scale y); loc is the CUMULATIVE displacement
    from the actor's last observed position, expressed in the actor's heading-aligned frame
    (it must be rotated back by rotate_angles). pi[N, 6] are mode logits (softmax -> probabilities).

Coordinate system used here (an explicit, documented approximation - NOT metric):

    local_x = (px - W/2) * k          k = IMAGE_WIDTH_M / W   ("pseudo-metres per pixel")
    local_y = (H - py)   * k          (image up = +y; origin = bottom-centre of the image)

The image is treated as a flat plane whose full width spans IMAGE_WIDTH_M pseudo-metres. Perspective is
ignored. Real metric motion would need camera calibration (ground-plane homography); monocular depth
alone has not been validated for that. The mapping is exactly invertible (see local_to_pixels).

Time handling: observations arrive at the detection rate (about 1 Hz on this CPU), not 10 Hz. HiVT needs a
uniform 0.1 s grid, so history is LINEARLY INTERPOLATED between consecutive real observations, only when
the real gap is <= max_gap_s, and every result reports how much of its history was interpolated. Nothing is
extrapolated, duplicated, or invented outside the span of real observations.

Written to be compatible with Python 3.8 (the HiVT repository targets 3.8).
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from itertools import permutations
from pathlib import Path

import numpy as np

HIVT_HISTORY_STEPS = 20
HIVT_FUTURE_STEPS = 30
HIVT_NUM_MODES = 6
DEFAULT_DT_S = 0.1
DEFAULT_IMAGE_WIDTH_M = 30.0
DEFAULT_MAX_GAP_S = 1.5
DEFAULT_MIN_OBS = 3
DEFAULT_MAX_AGENTS = 12
COORDINATE_SPACE = "image_plane_pseudo_metric_experimental_not_metric"

REASON_INSUFFICIENT_OBS = "insufficient_observations"
REASON_GAP = "gap_exceeds_limit"
REASON_NOT_CURRENT = "not_observed_in_current_frame"
REASON_OVER_CAP = "dropped_over_agent_cap"
REASON_BAD_INPUT = "invalid_input"


# --------------------------------------------------------------------------------------
# Coordinate transforms (pure numpy, exactly invertible)
# --------------------------------------------------------------------------------------
def _scale(image_size, width_m):
    w, h = float(image_size[0]), float(image_size[1])
    if not (w > 0 and h > 0 and width_m > 0):
        raise ValueError(f"invalid image size / scale: {image_size}, {width_m}")
    return w, h, width_m / w


def pixels_to_local(xy_px, image_size, width_m=DEFAULT_IMAGE_WIDTH_M):
    """[..., 2] pixel coordinates -> pseudo-metric scene coordinates."""
    w, h, k = _scale(image_size, width_m)
    xy = np.asarray(xy_px, dtype=np.float64)
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - w / 2.0) * k
    out[..., 1] = (h - xy[..., 1]) * k
    return out


def local_to_pixels(xy_local, image_size, width_m=DEFAULT_IMAGE_WIDTH_M):
    """Exact inverse of pixels_to_local."""
    w, h, k = _scale(image_size, width_m)
    xy = np.asarray(xy_local, dtype=np.float64)
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] / k + w / 2.0
    out[..., 1] = h - xy[..., 1] / k
    return out


def heading_to_world(displacement, angle):
    """Rotate [..., 2] displacements from an actor's heading-aligned frame back to the scene frame.

    HiVT trains with y_aligned = y_world @ R, R = [[c,-s],[s,c]], i.e. a rotation by -angle,
    so the inverse is a rotation by +angle.
    """
    c, s = math.cos(angle), math.sin(angle)
    d = np.asarray(displacement, dtype=np.float64)
    out = np.empty_like(d)
    out[..., 0] = d[..., 0] * c - d[..., 1] * s
    out[..., 1] = d[..., 0] * s + d[..., 1] * c
    return out


# --------------------------------------------------------------------------------------
# History bookkeeping (main process; numpy only)
# --------------------------------------------------------------------------------------
class HiVTTrackHistory:
    """Per-ByteTrack-ID rolling list of REAL (timestamp, x_px, y_px) observations.

    * Histories never mix between IDs.
    * An ID whose next observation arrives more than reset_gap_s late starts a NEW history (a long
      disappearance is not bridged), and is counted.
    * Timestamps must increase strictly per ID; out-of-order / duplicate ones are dropped and counted.
    * If the image size changes, every history is cleared (different camera/video geometry).
    * Nothing is interpolated here; interpolation happens only inside the predictor, on a copy.
    """

    def __init__(self, anchor="bottom", keep_s=8.0, reset_gap_s=3.0, max_obs=96, prune_after_s=6.0):
        if anchor not in ("bottom", "center"):
            raise ValueError("anchor must be 'bottom' or 'center'")
        self.anchor = anchor
        self.keep_s = float(keep_s)
        self.reset_gap_s = float(reset_gap_s)
        self.prune_after_s = float(prune_after_s)
        self.max_obs = int(max_obs)
        self.obs = {}              # track_id -> deque[(t, x, y)]
        self.image_size = None
        self.stats = {"obs_added": 0, "invalid_box": 0, "untracked_detection": 0,
                      "out_of_order": 0, "history_reset_gap": 0, "image_size_reset": 0,
                      "tracks_pruned": 0, "tracks_started": 0}

    def _anchor_of(self, box):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None
        return ((x1 + x2) * 0.5, y2 if self.anchor == "bottom" else (y1 + y2) * 0.5)

    def update(self, timestamp_s, boxes, track_ids, image_size):
        """Add this frame's observations. Returns the number of observations accepted."""
        t = float(timestamp_s)
        if not math.isfinite(t):
            self.stats["invalid_box"] += len(boxes or [])
            return 0
        size = (int(image_size[0]), int(image_size[1]))
        if self.image_size is not None and size != self.image_size:
            self.obs.clear()
            self.stats["image_size_reset"] += 1
        self.image_size = size
        added = 0
        for box, tid in zip(boxes or [], track_ids or []):
            if tid is None:
                self.stats["untracked_detection"] += 1
                continue
            if box is None or len(box) < 4:
                self.stats["invalid_box"] += 1
                continue
            anchor = self._anchor_of(box)
            if anchor is None:
                self.stats["invalid_box"] += 1
                continue
            tid = int(tid)
            history = self.obs.get(tid)
            if history and t <= history[-1][0]:
                self.stats["out_of_order"] += 1
                continue
            if history and t - history[-1][0] > self.reset_gap_s:
                history.clear()
                self.stats["history_reset_gap"] += 1
            if history is None or not history:
                if history is None:
                    history = self.obs[tid] = deque(maxlen=self.max_obs)
                self.stats["tracks_started"] += 1
            history.append((t, anchor[0], anchor[1]))
            added += 1
            self.stats["obs_added"] += 1
        # forget old observations, and tracks that have been gone for a while
        for tid in list(self.obs):
            history = self.obs[tid]
            while history and t - history[0][0] > self.keep_s:
                history.popleft()
            if not history or t - history[-1][0] > self.prune_after_s:
                del self.obs[tid]
                self.stats["tracks_pruned"] += 1
        self.stats["tracked_ids_now"] = len(self.obs)
        return added

    def snapshot(self, timestamp_s):
        """Payload for the worker: only tracks observed in THIS frame (last observation == timestamp_s)."""
        t = float(timestamp_s)
        tracks = []
        for tid, history in self.obs.items():
            if history and abs(history[-1][0] - t) < 1e-9:
                tracks.append({"track_id": int(tid),
                               "obs": [[round(a, 6), round(b, 2), round(c, 2)] for a, b, c in history]})
        return {"timestamp_s": t, "image_size": list(self.image_size) if self.image_size else None,
                "tracks": tracks}


# --------------------------------------------------------------------------------------
# Time-grid resampling and scene construction (pure numpy; unit-testable without torch)
# --------------------------------------------------------------------------------------
def build_agent_grid(obs, t_now, dt=DEFAULT_DT_S, steps=HIVT_HISTORY_STEPS,
                     max_gap_s=DEFAULT_MAX_GAP_S, min_obs=DEFAULT_MIN_OBS):
    """Resample one track's REAL observations onto HiVT's uniform history grid.

    Grid time of step k is t_now - (steps-1-k)*dt, so step steps-1 is "now" (a real observation).
    A grid point is
      * real          - coincides with a real observation (|dt| < 1e-6)
      * interpolated  - lies between two consecutive real observations at most max_gap_s apart
      * invalid       - before the first observation (track younger than the window) or inside a
                        longer gap. Invalid steps are marked in padding_mask, never filled.
    Returns (grid_dict, reason). reason is None when the track is acceptable.
    """
    arr = np.asarray(obs, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        return None, REASON_BAD_INPUT
    if np.any(np.diff(arr[:, 0]) <= 0):
        return None, REASON_BAD_INPUT
    t, xy = arr[:, 0], arr[:, 1:3]
    if abs(t[-1] - t_now) > 1e-6:
        return None, REASON_NOT_CURRENT
    grid_t = t_now - (steps - 1 - np.arange(steps)) * dt
    valid = np.zeros(steps, dtype=bool)
    real = np.zeros(steps, dtype=bool)
    out = np.zeros((steps, 2), dtype=np.float64)
    gap_violation = False
    for k, tk in enumerate(grid_t):
        if tk < t[0] - 1e-9:
            continue                                   # before the track existed: padded

        # Check both neighboring observations to handle floating-point rounding.
        right = int(np.searchsorted(t, tk, side="left"))

        if right < len(t) and abs(t[right] - tk) < 1e-6:
            valid[k] = real[k] = True
            out[k] = xy[right]
            continue

        if right > 0 and abs(t[right - 1] - tk) < 1e-6:
            valid[k] = real[k] = True
            out[k] = xy[right - 1]
            continue

        j = right - 1
        if j < 0 or j + 1 >= len(t):
            continue
        if t[j + 1] - t[j] > max_gap_s:
            gap_violation = True
            continue                                   # inside a gap that is too long: padded

        w = (tk - t[j]) / (t[j + 1] - t[j])
        out[k] = xy[j] * (1.0 - w) + xy[j + 1] * w
        valid[k] = True
    # number of REAL observations that actually support the window (incl. the one bracketing step 0)
    before = np.nonzero(t <= grid_t[0] + 1e-9)[0]
    first = int(before[-1]) if before.size else 0
    n_real_used = int(len(t) - first)
    if n_real_used < min_obs:
        return None, REASON_INSUFFICIENT_OBS
    if gap_violation:
        return None, REASON_GAP
    n_valid = int(valid.sum())
    n_interp = int(valid.sum() - real.sum())
    grid = {"valid": valid, "real": real, "xy_px": out, "grid_t": grid_t,
            "n_real_used": n_real_used, "n_valid_steps": n_valid,
            "interpolated_fraction": (n_interp / n_valid) if n_valid else 1.0}
    return grid, None


def build_scene(tracks, image_size, t_now, dt=DEFAULT_DT_S, width_m=DEFAULT_IMAGE_WIDTH_M,
                max_gap_s=DEFAULT_MAX_GAP_S, min_obs=DEFAULT_MIN_OBS, max_agents=DEFAULT_MAX_AGENTS,
                history_steps=HIVT_HISTORY_STEPS, future_steps=HIVT_FUTURE_STEPS):
    """Build the exact arrays HiVT expects. Returns (scene_or_None, rejected{track_id: reason})."""
    rejected, accepted = {}, []
    for tr in tracks or []:
        tid = int(tr["track_id"])
        try:
            grid, reason = build_agent_grid(tr["obs"], t_now, dt, history_steps, max_gap_s, min_obs)
        except Exception:                                  # malformed payload for this track only
            grid, reason = None, REASON_BAD_INPUT
        if reason is not None:
            rejected[tid] = reason
        else:
            accepted.append((tid, grid))
    # keep the closest-to-camera vehicles (largest y_px = lowest in the image) if over the cap
    accepted.sort(key=lambda item: -item[1]["xy_px"][history_steps - 1, 1])
    for tid, _ in accepted[max_agents:]:
        rejected[tid] = REASON_OVER_CAP
    accepted = accepted[:max_agents]
    if not accepted:
        return None, rejected

    n, total = len(accepted), history_steps + future_steps
    positions = np.zeros((n, total, 2), dtype=np.float32)
    padding = np.ones((n, total), dtype=bool)
    x = np.zeros((n, history_steps, 2), dtype=np.float32)
    rotate = np.zeros((n,), dtype=np.float32)
    ids = []
    for i, (tid, g) in enumerate(accepted):
        ids.append(tid)
        local = pixels_to_local(g["xy_px"], image_size, width_m)
        v = g["valid"]
        positions[i, :history_steps][v] = local[v]
        padding[i, :history_steps] = ~v
        valid_steps = np.nonzero(v)[0]
        # displacements exactly as in the dataset: zero unless both steps are valid
        for t in range(1, history_steps):
            if v[t] and v[t - 1]:
                x[i, t] = positions[i, t, :] - positions[i, t - 1, :]
        if len(valid_steps) > 1:                           # heading from the last two valid steps
            hv = positions[i, valid_steps[-1]] - positions[i, valid_steps[-2]]
            rotate[i] = math.atan2(float(hv[1]), float(hv[0]))
    bos = np.zeros((n, history_steps), dtype=bool)
    bos[:, 0] = ~padding[:, 0]
    bos[:, 1:] = padding[:, :history_steps - 1] & ~padding[:, 1:history_steps]
    pairs = list(permutations(range(n), 2))
    edge_index = (np.asarray(pairs, dtype=np.int64).T.copy() if pairs
                  else np.zeros((2, 0), dtype=np.int64))
    scene = {"track_ids": ids, "x": x, "positions": positions, "padding_mask": padding,
             "bos_mask": bos, "rotate_angles": rotate, "edge_index": edge_index,
             "grids": [g for _, g in accepted]}
    return scene, rejected


# --------------------------------------------------------------------------------------
# Model wrapper (worker process; needs torch + HiVT repo)
# --------------------------------------------------------------------------------------
def default_checkpoint(repo_dir, variant="64"):
    return (Path(repo_dir) / "checkpoints" / f"HiVT-{variant}" / "checkpoints"
            / "epoch=63-step=411903.ckpt")


def _apply_torch_compat():
    """Make HiVT's TemporalEncoderLayer usable with PyTorch >= 2.0 (no-op on older versions).

    nn.TransformerEncoder in PyTorch >= 2.0 calls layer(..., is_causal=...). HiVT's custom layer
    predates that argument. HiVT already passes its causal mask explicitly (mask=attn_mask), so the
    extra hint is redundant and is simply accepted and ignored. Applied only if the installed layer
    does not already accept it; the HiVT source files on disk are never modified.
    """
    import inspect
    from models import local_encoder as local_encoder_module
    layer_cls = getattr(local_encoder_module, "TemporalEncoderLayer", None)
    if layer_cls is None or "is_causal" in inspect.signature(layer_cls.forward).parameters:
        return False
    original = layer_cls.forward

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        return original(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    layer_cls.forward = forward
    return True


class HiVTPredictor:
    def __init__(self, repo_dir, checkpoint_path=None, variant="64", device="cpu",
                 dt=DEFAULT_DT_S, width_m=DEFAULT_IMAGE_WIDTH_M, max_gap_s=DEFAULT_MAX_GAP_S,
                 min_obs=DEFAULT_MIN_OBS, max_agents=DEFAULT_MAX_AGENTS, threads=None):
        import torch                                       # imported here: main process never needs it
        self.torch = torch
        self.repo_dir = Path(repo_dir).resolve()
        self.variant = str(variant)
        self.device = torch.device(device)
        self.dt, self.width_m = float(dt), float(width_m)
        self.max_gap_s, self.min_obs, self.max_agents = float(max_gap_s), int(min_obs), int(max_agents)
        if threads:
            torch.set_num_threads(int(threads))
        # The repo must win the import race against any other 'models' / 'utils' on sys.path.
        if str(self.repo_dir) in sys.path:
            sys.path.remove(str(self.repo_dir))
        sys.path.insert(0, str(self.repo_dir))
        from models.hivt import HiVT
        from utils import TemporalData
        for mod_name in ("models.hivt", "utils"):
            mod_file = Path(sys.modules[mod_name].__file__).resolve()
            if self.repo_dir not in mod_file.parents:
                raise ImportError(f"'{mod_name}' was imported from {mod_file}, not from the HiVT repo "
                                  f"{self.repo_dir}; another module with the same name shadows it")
        self._TemporalData = TemporalData
        self.torch_compat_shim = _apply_torch_compat()
        ckpt = Path(checkpoint_path) if checkpoint_path else default_checkpoint(self.repo_dir, self.variant)
        if not ckpt.is_file():
            raise FileNotFoundError(f"HiVT checkpoint not found: {ckpt}")
        self.checkpoint = ckpt
        t0 = time.perf_counter()
        try:
            self.model = HiVT.load_from_checkpoint(checkpoint_path=str(ckpt), map_location=self.device)
        except Exception as exc:  # noqa: BLE001
            # PyTorch >= 2.6 defaults to weights_only=True, which rejects Lightning checkpoints that
            # embed callback objects. Older PyTorch (the one this checkpoint was made for) does not
            # have that restriction. Retry ONLY for that specific error, on this local checkpoint.
            if "weights_only" not in str(exc) and "Unsupported global" not in str(exc):
                raise
            self.model = HiVT.load_from_checkpoint(checkpoint_path=str(ckpt), map_location=self.device,
                                                   weights_only=False)
        self.model.eval()
        self.model.freeze()                                # requires_grad False + eval()
        self.load_s = time.perf_counter() - t0
        hp = self.model.hparams
        self.history_steps = int(hp.historical_steps)
        self.future_steps = int(hp.future_steps)
        self.num_modes = int(hp.num_modes)
        self.rotate = bool(hp.rotate)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if self.history_steps != HIVT_HISTORY_STEPS or self.future_steps != HIVT_FUTURE_STEPS:
            raise RuntimeError(
                f"checkpoint config ({self.history_steps} history / {self.future_steps} future steps) "
                f"differs from what the adapter was written for "
                f"({HIVT_HISTORY_STEPS}/{HIVT_FUTURE_STEPS})")

    def info(self):
        hp = self.model.hparams
        return {"checkpoint": str(self.checkpoint), "torch": self.torch.__version__,
                "parameters": int(self.n_params), "historical_steps": self.history_steps,
                "future_steps": self.future_steps, "num_modes": self.num_modes,
                "embed_dim": int(hp.embed_dim), "num_heads": int(hp.num_heads),
                "num_temporal_layers": int(hp.num_temporal_layers),
                "num_global_layers": int(hp.num_global_layers),
                "local_radius": float(hp.local_radius), "rotate": self.rotate,
                "training": bool(self.model.training), "torch_compat_shim": bool(self.torch_compat_shim)}

    def _to_data(self, scene):
        torch = self.torch
        n = len(scene["track_ids"])
        empty = lambda *shape, dtype=torch.float32: torch.empty(shape, dtype=dtype)  # noqa: E731
        return self._TemporalData(
            x=torch.from_numpy(scene["x"]),
            positions=torch.from_numpy(scene["positions"]),
            edge_index=torch.from_numpy(scene["edge_index"]),
            padding_mask=torch.from_numpy(scene["padding_mask"]),
            bos_mask=torch.from_numpy(scene["bos_mask"]),
            rotate_angles=torch.from_numpy(scene["rotate_angles"]),
            # No HD map from a monocular camera: empty lane context (structurally valid).
            lane_vectors=empty(0, 2),
            is_intersections=empty(0, dtype=torch.uint8),
            turn_directions=empty(0, dtype=torch.uint8),
            traffic_controls=empty(0, dtype=torch.uint8),
            lane_actor_index=empty(2, 0, dtype=torch.long),
            lane_actor_vectors=empty(0, 2),
            num_nodes=n,
        ).to(self.device)

    def predict(self, tracks, image_size, t_now):
        """Forecast every acceptable track. Returns (predictions, timing_ms).

        predictions: one dict per input track (valid=True with modes, or valid=False with a reason).
        """
        torch = self.torch
        t_build = time.perf_counter()
        scene, rejected = build_scene(tracks, image_size, float(t_now), self.dt, self.width_m,
                                      self.max_gap_s, self.min_obs, self.max_agents,
                                      self.history_steps, self.future_steps)
        results = [{"track_id": int(tid), "valid": False, "reason": reason}
                   for tid, reason in rejected.items()]
        timing = {"build_ms": 0.0, "infer_ms": 0.0}
        if scene is not None:
            data = self._to_data(scene)
            timing["build_ms"] = (time.perf_counter() - t_build) * 1000.0
            t_inf = time.perf_counter()
            with torch.no_grad():
                y_hat, pi = self.model(data)
            timing["infer_ms"] = (time.perf_counter() - t_inf) * 1000.0
            n = len(scene["track_ids"])
            expected = (self.num_modes, n, self.future_steps, 4)
            if tuple(y_hat.shape) != expected or tuple(pi.shape) != (n, self.num_modes):
                raise RuntimeError(f"unexpected HiVT output shapes: y_hat {tuple(y_hat.shape)} "
                                   f"(expected {expected}), pi {tuple(pi.shape)}")
            loc = y_hat[..., :2].detach().cpu().numpy().astype(np.float64)       # [F,N,H,2]
            probs = torch.softmax(pi, dim=-1).detach().cpu().numpy().astype(np.float64)
            if not (np.all(np.isfinite(loc)) and np.all(np.isfinite(probs))):
                raise RuntimeError("HiVT produced NaN/Inf; predictions discarded")
            for i, tid in enumerate(scene["track_ids"]):
                g = scene["grids"][i]
                angle = float(scene["rotate_angles"][i]) if self.rotate else 0.0
                last_local = scene["positions"][i, self.history_steps - 1].astype(np.float64)
                modes = []
                for m in np.argsort(-probs[i]):
                    disp = heading_to_world(loc[m, i], angle)                    # cumulative displacement
                    pts = local_to_pixels(last_local + disp, image_size, self.width_m)
                    modes.append({"mode": int(m), "probability": float(probs[i, m]),
                                  "points": [[round(float(a), 2), round(float(b), 2)] for a, b in pts]})
                last_px = g["xy_px"][self.history_steps - 1]
                results.append({
                    "track_id": int(tid), "valid": True, "model": f"HiVT-{self.variant}",
                    "coordinate_space": COORDINATE_SPACE,
                    "t_obs": float(t_now), "dt_s": self.dt,
                    "horizon_s": self.dt * self.future_steps,
                    "anchor_px": [round(float(last_px[0]), 2), round(float(last_px[1]), 2)],
                    "real_obs_used": int(g["n_real_used"]),
                    "history_valid_steps": int(g["n_valid_steps"]),
                    "interpolated_fraction": round(float(g["interpolated_fraction"]), 3),
                    "lanes_used": False,
                    "modes": modes,
                })
        return results, timing
