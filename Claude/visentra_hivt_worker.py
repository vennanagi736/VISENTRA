"""
Persistent HiVT worker. Run with the dedicated HiVT environment:

    V:\\visentra\\research_models\\HiVT\\.venv\\Scripts\\python.exe visentra_hivt_worker.py [options]

Protocol (one JSON object per line, same shape as VISENTRA's other workers):
  worker -> parent   {"ready": true, "model": "HiVT-64", "load_s": .., "warmup_s": .., "info": {...}}
                     {"ready": false, "error": "..."}                      then exit code 2
  parent -> worker   {"frame_id": int, "timestamp_s": float, "image_size": [W, H],
                      "tracks": [{"track_id": int, "obs": [[t, x_px, y_px], ...]}, ...]}
  worker -> parent   {"frame_id": int, "timestamp_s": float, "predictions": [...],
                      "build_ms": .., "infer_ms": .., "total_ms": ..}
                     {"frame_id": int, "error": "..."}                     (this request failed, worker lives)
                     {"frame_id": int, "fatal": true, "error": "..."}      (worker is about to exit)
  parent -> worker   {"command": "quit"}   or simply close stdin (clean exit code 0)

The model is loaded once, put in eval mode with gradients off, and runs on CPU. The worker keeps NO
per-track state: the parent sends each track's real observation window with every request, so a
dropped request can never corrupt a history. stdout carries protocol lines only; anything printed by
libraries is routed to stderr.
"""
import argparse
import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

_PROTOCOL = sys.stdout            # the real stdout, used only for protocol lines
sys.stdout = sys.stderr           # stray prints from torch / lightning / PyG must not corrupt the protocol
for _stream in (sys.stdin, _PROTOCOL, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:             # noqa: BLE001
        pass

_LOG = None
_log_path = os.environ.get("VISENTRA_WORKER_LOG")
if _log_path:
    try:
        _LOG = open(_log_path, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_LOG)
    except OSError:
        _LOG = None


def wlog(message):
    line = time.strftime("%H:%M:%S ") + message
    if _LOG is not None:
        _LOG.write(line + "\n")
    print("[hivt-worker] " + message, file=sys.stderr, flush=True)


def send(obj):
    _PROTOCOL.write(json.dumps(obj, separators=(",", ":")) + "\n")
    _PROTOCOL.flush()


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(here / "research_models" / "HiVT"))
    p.add_argument("--variant", choices=["64", "128"], default="64")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--image-width-m", type=float, default=30.0)
    p.add_argument("--max-gap", type=float, default=1.5)
    p.add_argument("--min-obs", type=int, default=3)
    p.add_argument("--max-agents", type=int, default=12)
    p.add_argument("--threads", type=int, default=int(os.environ.get("VISENTRA_TORCH_THREADS", "2")))
    return p.parse_args()


def warm_up(predictor):
    """Run one synthetic, structurally valid scene so structural problems fail at start-up, not later."""
    t_now, w, h = 100.0, 640, 480
    tracks = []
    for tid, (x0, y0, vx, vy) in enumerate([(200, 400, 30, -20), (420, 380, -25, -15)], start=1):
        tracks.append({"track_id": tid,
                       "obs": [[t_now - 2.0 + 1.0 * i, x0 + vx * i, y0 + vy * i] for i in range(3)]})
    t0 = time.perf_counter()
    preds, _ = predictor.predict(tracks, (w, h), t_now)
    if not any(p.get("valid") for p in preds):
        raise RuntimeError(f"warm-up produced no valid prediction: {preds}")
    return time.perf_counter() - t0


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from visentra_hivt_adapter import HiVTPredictor
        t0 = time.perf_counter()
        predictor = HiVTPredictor(
            args.repo, checkpoint_path=args.checkpoint, variant=args.variant, device="cpu",
            dt=args.dt, width_m=args.image_width_m, max_gap_s=args.max_gap,
            min_obs=args.min_obs, max_agents=args.max_agents, threads=args.threads)
        load_s = time.perf_counter() - t0
        warmup_s = warm_up(predictor)
        info = predictor.info()
        wlog(f"HiVT-{args.variant} loaded in {load_s:.1f}s, warm-up {warmup_s:.2f}s, "
             f"{info['parameters']:,} parameters, threads {args.threads}")
        send({"ready": True, "model": f"HiVT-{args.variant}", "load_s": round(load_s, 2),
              "warmup_s": round(warmup_s, 3), "info": info})
    except BaseException as exc:      # noqa: BLE001
        wlog("start-up failed:\n" + traceback.format_exc())
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        raise SystemExit(2)

    frame_id = None
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            frame_id = None
            try:
                req = json.loads(line)
                if req.get("command") == "quit":
                    break
                frame_id = req.get("frame_id")
                t_start = time.perf_counter()
                preds, timing = predictor.predict(req["tracks"], req["image_size"], req["timestamp_s"])
                send({"frame_id": frame_id, "timestamp_s": req["timestamp_s"], "predictions": preds,
                      "build_ms": round(timing["build_ms"], 2), "infer_ms": round(timing["infer_ms"], 2),
                      "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2)})
            except Exception as exc:  # noqa: BLE001  one bad request must not kill the worker
                wlog(f"request {frame_id} failed:\n{traceback.format_exc()}")
                send({"frame_id": frame_id, "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:      # noqa: BLE001  e.g. SystemExit/KeyboardInterrupt: say so, then exit
        wlog("fatal:\n" + traceback.format_exc())
        try:
            send({"frame_id": frame_id, "fatal": True, "error": f"{type(exc).__name__}: {exc}"})
        except Exception:             # noqa: BLE001
            pass
        raise
    wlog("stdin closed / quit received: exiting cleanly")


if __name__ == "__main__":
    main()
