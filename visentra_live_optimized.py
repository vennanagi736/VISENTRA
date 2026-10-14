import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# VISENTRA PATHS
# ============================================================
ROOT = Path(r"V:\visentra")
RF_PYTHON = ROOT / r"environments\rfdetr_env\Scripts\python.exe"
LANE_PYTHON = ROOT / r"environments\condlanenet_env\Scripts\python.exe"
RF_MODEL = ROOT / r"models\rf-detr-medium.pth"
LANE_MODEL = ROOT / r"models\curvelanes_medium.pth"
COND_LANE_ROOT = ROOT / "CondLaneNet"
LANE_CONFIG = COND_LANE_ROOT / r"configs\condlanenet\curvelanes\curvelanes_medium_test.py"

# Metric3D runs in the existing MonoPilot environment, not the RF-DETR env.
METRIC_PYTHON = ROOT / r"MonoPilot\.venv\Scripts\python.exe"
METRIC_MODEL = ROOT / r"metric3d_test\models\metric3d_v2_vit_small.onnx"
METRIC_CONFIG = ROOT / r"metric3d_test\models\preprocessor_config.json"
METRIC_RESULTS_DIR = ROOT / r"metric3d_test\results"

OUTPUT_DIR = ROOT / "outputs"
TMP_DIR = ROOT / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
METRIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)
tempfile.tempdir = str(TMP_DIR)

# Provisional empirically-fitted factor from the initial 5/10/15/20 m samples.
# It is NOT an independently validated camera calibration.
DEFAULT_DEPTH_SCALE = 0.373572
DEPTH_CAP_VALUE = 199.5

# Only estimate/display distance for road-vehicle classes. Other RF-DETR
# detections may still be drawn, but they are not treated as vehicle targets.
VEHICLE_CLASS_NAMES = {
    "car", "truck", "bus", "motorcycle", "motorbike",
    "bicycle", "bike", "van", "vehicle", "train", "tram",
}


def is_vehicle_class(class_name):
    """Return True only for labels considered vehicles by this demo."""
    normalized = str(class_name or "").strip().lower().replace("_", " ")
    return normalized in VEHICLE_CLASS_NAMES


_RF_DETECTOR = None
_LANE_DETECTOR = None
_LANE_CFG = None
_METRIC_ESTIMATOR = None


def check_paths(require_lane=True):
    paths = {
        "RF-DETR Python": RF_PYTHON,
        "RF-DETR model": RF_MODEL,
        "Metric3D Python": METRIC_PYTHON,
        "Metric3D ONNX model": METRIC_MODEL,
        "Metric3D preprocessing config": METRIC_CONFIG,
    }
    if require_lane:
        paths.update({
            "CurveLaneS Python": LANE_PYTHON,
            "CurveLaneS model": LANE_MODEL,
            "CurveLaneS config": LANE_CONFIG,
            "CondLaneNet source": COND_LANE_ROOT,
        })
    print("\n========== VISENTRA PATH CHECK ==========")
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        print(f"[OK] {name}: {path}")
    print("=========================================\n")


# ============================================================
# RF-DETR VEHICLE / OBJECT DETECTION
# ============================================================
def run_rfdetr(image_path):
    """Run RF-DETR and return JSON-safe boxes, classes, names and confidences."""
    global _RF_DETECTOR
    if _RF_DETECTOR is None:
        print("[RF-DETR] Loading checkpoint once...")
        from rfdetr import RFDETRBase
        _RF_DETECTOR = RFDETRBase.from_checkpoint(str(RF_MODEL))
        print("[RF-DETR] Model loaded.")

    detections = _RF_DETECTOR.predict(str(image_path), threshold=0.40)
    result = {"boxes": [], "class_ids": [], "class_names": [], "confidences": []}
    names = detections.data.get("class_name") if getattr(detections, "data", None) else None

    for index, (box, class_id, confidence) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.confidence
    )):
        result["boxes"].append([float(v) for v in box])
        try:
            result["class_ids"].append(int(class_id))
        except (TypeError, ValueError):
            result["class_ids"].append(-1)
        result["confidences"].append(float(confidence))
        if names is not None and index < len(names):
            result["class_names"].append(str(names[index]))
        else:
            result["class_names"].append(str(class_id))
    return result


# ============================================================
# CURVELANES MODEL LOADING
# ============================================================
def load_curvelanes():
    print("[CurveLaneS] Loading checkpoint once...")
    cond_root = str(COND_LANE_ROOT)
    if cond_root not in sys.path:
        sys.path.insert(0, cond_root)

    import torch
    if not torch.cuda.is_available():
        # Compatibility patch retained for this CPU-only environment.
        torch.cuda.FloatTensor = torch.FloatTensor

    import mmcv
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    config_namespace = {"__file__": str(LANE_CONFIG), "__name__": "__curvelanes_config__"}
    with open(LANE_CONFIG, "r", encoding="utf-8") as handle:
        config_source = handle.read()
    exec(compile(config_source, str(LANE_CONFIG), "exec"), config_namespace)
    cfg_dict = {key: value for key, value in config_namespace.items() if not key.startswith("__")}
    cfg = mmcv.Config(cfg_dict)
    cfg.model.head.location_configs.device = "cpu"
    cfg.model.pretrained = None

    model = build_detector(cfg.model)
    load_checkpoint(model, str(LANE_MODEL), map_location="cpu")
    model = model.cpu()
    model.eval()
    print("[CurveLaneS] Model loaded.")
    return model, cfg


def prepare_curvelanes_image(image_path, cfg):
    from mmdet.datasets.pipelines import Compose
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_shape = image.shape
    results = {
        "filename": str(image_path),
        "sub_img_name": Path(image_path).name,
        "img": image,
        "gt_points": [],
        "id_classes": [],
        "id_instances": [],
        "img_shape": original_shape,
        "ori_shape": original_shape,
        "crop_offset": [0, 0],
        "crop_shape": original_shape,
    }
    pipeline = Compose(cfg.val_pipeline)
    data = pipeline(results)
    if data is None:
        raise RuntimeError("CurveLaneS preprocessing returned None.")
    return data, original_shape


def unwrap_data_container(value):
    if not hasattr(value, "data"):
        return value
    value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def prepare_model_inputs(data):
    import torch
    img = unwrap_data_container(data["img"])
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    img = img.float()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() != 4:
        raise RuntimeError(f"Unexpected CurveLaneS image tensor shape: {tuple(img.shape)}")
    img = img.cpu()

    meta_container = data["img_metas"]
    meta = meta_container.data if hasattr(meta_container, "data") else meta_container
    while isinstance(meta, (list, tuple)) and len(meta) == 1:
        meta = meta[0]
    if isinstance(meta, (list, tuple)):
        if not meta:
            raise RuntimeError("CurveLaneS img_metas is empty.")
        if isinstance(meta[0], dict):
            meta = meta[0]
    if not isinstance(meta, dict):
        raise RuntimeError(f"Could not extract CurveLaneS metadata; got {type(meta).__name__}")
    return img, [meta], meta


# ============================================================
# CURVELANES INFERENCE
# ============================================================
def run_curvelanes(image_path):
    global _LANE_DETECTOR, _LANE_CFG
    cond_root = str(COND_LANE_ROOT)
    if cond_root not in sys.path:
        sys.path.insert(0, cond_root)
    from tools.condlanenet.post_process import CurvelanesPostProcessor

    if _LANE_DETECTOR is None:
        _LANE_DETECTOR, _LANE_CFG = load_curvelanes()
    model, cfg = _LANE_DETECTOR, _LANE_CFG

    data, original_shape = prepare_curvelanes_image(image_path, cfg)
    img, img_metas, meta = prepare_model_inputs(data)
    import torch
    with torch.no_grad():
        seeds, _ = model(
            img=img,
            img_metas=img_metas,
            return_loss=False,
            rescale=False,
            thr=0.3,
        )

    post_processor = CurvelanesPostProcessor(mask_size=cfg.mask_size, hm_thr=0.3)
    lanes, _ = post_processor(seeds, meta["down_scale"])
    processed_shape = meta["img_shape"]
    h_processed, w_processed = processed_shape[0], processed_shape[1]
    h_original, w_original = original_shape[0], original_shape[1]
    ratio_x = w_original / w_processed
    ratio_y = h_original / h_processed

    final_lanes = []
    if lanes is not None:
        for lane in lanes:
            points = []
            for point in lane.get("points", []):
                x = max(0.0, min(float(point[0]) * ratio_x, w_original - 1))
                y = max(0.0, min(float(point[1]) * ratio_y, h_original - 1))
                points.append([int(round(x)), int(round(y))])
            if len(points) > 1:
                final_lanes.append(points)
    return final_lanes


# ============================================================
# METRIC3D V2 DEPTH ESTIMATION
# ============================================================
class Metric3DEstimator:
    """Persistent CPU ONNX Runtime wrapper for the downloaded Metric3D v2 export."""

    def __init__(self):
        import onnxruntime as ort
        from PIL import Image  # noqa: F401 - validates dependency early

        if not METRIC_MODEL.is_file():
            raise FileNotFoundError(f"Metric3D model not found: {METRIC_MODEL}")
        if not METRIC_CONFIG.is_file():
            raise FileNotFoundError(f"Metric3D config not found: {METRIC_CONFIG}")

        with METRIC_CONFIG.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(os.environ.get("VISENTRA_ORT_THREADS", "3")))
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(METRIC_MODEL),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        names = [item.name.lower() for item in self.session.get_outputs()]
        self.depth_output_index = next(
            (index for index, name in enumerate(names) if "depth" in name), None
        )
        if self.depth_output_index is None:
            raise RuntimeError(f"Could not find depth output in: {names}")
        print("[Metric3D] ONNX model loaded on CPU.")
        print(f"[Metric3D] Input: {self.input_name}; outputs: {names}")

    def predict(self, image_path):
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise RuntimeError(f"Invalid image dimensions: {image_path}")

        config = self.config
        target_h = int(config.get("size", {}).get("height", 518))
        target_w = int(config.get("size", {}).get("width", 518))
        divisor = max(1, int(config.get("ensure_multiple_of", 14)))

        if config.get("do_resize", True):
            if config.get("keep_aspect_ratio", True):
                scale = min(target_w / original_width, target_h / original_height)
                resized_w = max(
                    divisor,
                    min(target_w, int(round(original_width * scale / divisor)) * divisor),
                )
                resized_h = max(
                    divisor,
                    min(target_h, int(round(original_height * scale / divisor)) * divisor),
                )
            else:
                resized_w, resized_h = target_w, target_h
            image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

        rgb = np.asarray(image, dtype=np.float32)
        # Match the downloaded preprocessor_config.json: no rescale or normalization.
        tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: tensor})
        depth = np.squeeze(np.asarray(outputs[self.depth_output_index], dtype=np.float32))
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected Metric3D depth tensor shape: {depth.shape}")
        if not np.isfinite(depth).any():
            raise RuntimeError("Metric3D depth output contains no finite values.")
        return depth


def estimate_box_distances(detections, depth_map, image_shape, depth_scale):
    """Sample central box regions from the full-image depth map; retain box ordering."""
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    depth_h, depth_w = depth_map.shape[:2]
    boxes = detections.get("boxes", []) or []
    class_ids = detections.get("class_ids", []) or []
    class_names = detections.get("class_names", []) or []
    confidences = detections.get("confidences", []) or []
    distance_results = []

    for index, box in enumerate(boxes):
        item = {
            "detection_index": int(index),
            "class_id": class_ids[index] if index < len(class_ids) else -1,
            "class_name": class_names[index] if index < len(class_names) else "object",
            "confidence": float(confidences[index]) if index < len(confidences) else None,
            "box_xyxy": [float(value) for value in box],
            "distance_m": None,
            "raw_depth_median": None,
            "raw_depth_p25": None,
            "raw_depth_p75": None,
            "depth_spread_m": None,
            "valid_pixel_fraction": 0.0,
            "capped_pixel_fraction": 0.0,
            "sample_pixels": 0,
            "status": "invalid_box",
        }
        # Keep detections such as people/benches available for drawing, but
        # never calculate or publish them as vehicle distances.
        if not is_vehicle_class(item["class_name"]):
            item["status"] = "non_vehicle_skipped"
            distance_results.append(item)
            continue

        if len(box) < 4:
            distance_results.append(item)
            continue

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0.0, min(x1, image_w - 1.0))
        x2 = max(0.0, min(x2, image_w - 1.0))
        y1 = max(0.0, min(y1, image_h - 1.0))
        y2 = max(0.0, min(y2, image_h - 1.0))
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 2 or box_h < 2:
            distance_results.append(item)
            continue

        # Sample the interior of the detection to reduce edge/background contamination.
        # If this interior is too small, fall back to the full detection box.
        def crop_to_depth(xa, ya, xb, yb):
            dx1 = max(0, min(depth_w - 1, int(np.floor(xa * depth_w / image_w))))
            dy1 = max(0, min(depth_h - 1, int(np.floor(ya * depth_h / image_h))))
            dx2 = max(dx1 + 1, min(depth_w, int(np.ceil(xb * depth_w / image_w))))
            dy2 = max(dy1 + 1, min(depth_h, int(np.ceil(yb * depth_h / image_h))))
            return depth_map[dy1:dy2, dx1:dx2]

        inner = crop_to_depth(
            x1 + 0.20 * box_w,
            y1 + 0.20 * box_h,
            x2 - 0.20 * box_w,
            y2 - 0.10 * box_h,
        )
        if inner.size < 4:
            inner = crop_to_depth(x1, y1, x2, y2)

        finite = np.isfinite(inner)
        capped_fraction = float(np.mean(finite & (inner >= DEPTH_CAP_VALUE))) if inner.size else 0.0
        valid_mask = finite & (inner > 0.0) & (inner < DEPTH_CAP_VALUE)
        values = inner[valid_mask]
        item["sample_pixels"] = int(inner.size)
        item["valid_pixel_fraction"] = float(values.size / inner.size) if inner.size else 0.0
        item["capped_pixel_fraction"] = capped_fraction

        if values.size < 2:
            item["status"] = "insufficient_uncapped_depth_pixels"
            distance_results.append(item)
            continue

        p25, median, p75 = [float(v) for v in np.percentile(values, [25, 50, 75])]
        item.update({
            "raw_depth_median": median,
            "raw_depth_p25": p25,
            "raw_depth_p75": p75,
            "distance_m": median * float(depth_scale),
            "depth_spread_m": (p75 - p25) * float(depth_scale),
            "status": "experimental_estimate",
        })
        distance_results.append(item)

    return distance_results


def get_metric_estimator():
    global _METRIC_ESTIMATOR
    if _METRIC_ESTIMATOR is None:
        _METRIC_ESTIMATOR = Metric3DEstimator()
    return _METRIC_ESTIMATOR


def run_metric3d(image_path, detections, depth_scale, save_artifacts=False):
    estimator = get_metric_estimator()
    depth_map = estimator.predict(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for Metric3D: {image_path}")
    distances = estimate_box_distances(detections, depth_map, image.shape, depth_scale)

    result = {
        "distances": distances,
        "depth_shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        "depth_scale": float(depth_scale),
        "calibration_status": "provisional_not_independently_validated",
    }
    if save_artifacts:
        stem = Path(image_path).stem
        raw_path = METRIC_RESULTS_DIR / f"visentra_{stem}_raw_depth.npy"
        viz_path = METRIC_RESULTS_DIR / f"visentra_{stem}_depth_visualization.png"
        np.save(raw_path, depth_map)
        finite = np.isfinite(depth_map)
        display = np.zeros(depth_map.shape, dtype=np.uint8)
        if finite.any():
            lo, hi = np.percentile(depth_map[finite], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            display[finite] = np.clip(
                (depth_map[finite] - lo) * 255.0 / (hi - lo), 0, 255
            ).astype(np.uint8)
        heatmap = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(viz_path), heatmap)
        result["raw_depth_path"] = str(raw_path)
        result["depth_visualization_path"] = str(viz_path)
    return result


# ============================================================
# DRAWING
# ============================================================
def select_two_lane_boundaries(lanes, image_shape):
    """Display heuristic: choose closest predicted curve on either side of centre."""
    height, width = image_shape[:2]
    centre_x = width / 2.0
    candidates = []
    for lane in lanes or []:
        if len(lane) < 2:
            continue
        lower = [point for point in lane if point[1] >= height * 0.55]
        points = lower if lower else lane
        representative_x = float(np.mean([point[0] for point in points]))
        candidates.append((representative_x, lane))

    left = [item for item in candidates if item[0] < centre_x]
    right = [item for item in candidates if item[0] >= centre_x]
    selected = []
    if left:
        selected.append(max(left, key=lambda item: item[0])[1])
    if right:
        selected.append(min(right, key=lambda item: item[0])[1])
    return selected


def draw_results(
    image_or_path,
    detections,
    lanes,
    output_path=None,
    fps=None,
    frame_number=None,
    distances=None,
    depth_scale=DEFAULT_DEPTH_SCALE,
    status_lines=None,
):
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path))
    else:
        image = image_or_path.copy()
    if image is None:
        raise RuntimeError(f"Could not read image: {image_or_path}")

    vehicle_color = (0, 180, 0)
    lane_colors = [(0, 220, 255), (255, 160, 0)]
    boxes = detections.get("boxes", [])
    class_ids = detections.get("class_ids", [])
    class_names = detections.get("class_names", [])
    confidences = detections.get("confidences", [])
    distance_by_index = {}
    if distances is not None:
        distance_by_index = {
            int(entry.get("detection_index", -1)): entry
            for entry in distances
        }

    for index, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        label = class_names[index] if index < len(class_names) else f"class {class_id}"
        distance_entry = distance_by_index.get(index)
        if is_vehicle_class(label):
            if distance_entry and distance_entry.get("distance_m") is not None:
                label += f" | ~{float(distance_entry['distance_m']):.1f} m"
            elif distances is not None:
                label += " | distance unavailable"
        cv2.rectangle(image, (x1, y1), (x2, y2), vehicle_color, 2)
        cv2.putText(
            image, label[:100], (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, vehicle_color, 2, cv2.LINE_AA,
        )

    image_centre_x = image.shape[1] / 2.0
    for lane in lanes or []:
        if len(lane) < 2:
            continue
        lower_points = [point for point in lane if point[1] >= image.shape[0] * 0.55]
        representative_points = lower_points if lower_points else lane
        representative_x = float(np.mean([point[0] for point in representative_points]))
        is_left = representative_x < image_centre_x
        color = lane_colors[0] if is_left else lane_colors[1]
        points = np.asarray(lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, color, 5, cv2.LINE_AA)
        label_x, label_y = lane[len(lane) // 2]
        cv2.putText(
            image, "Left boundary" if is_left else "Right boundary",
            (int(label_x), max(24, int(label_y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    status = f"Lane boundaries: {len(lanes or [])}/2"
    if fps is not None:
        status += f" | display FPS: {fps:.2f}"
    if frame_number is not None:
        status += f" | frame: {frame_number}"
    cv2.putText(image, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA)
    lines = list(status_lines or [])
    lines.append(f"Metric3D experimental scale: {float(depth_scale):.6f} (not safety-validated)")
    for offset, text in enumerate(lines, start=1):
        cv2.putText(image, str(text)[:140], (12, 28 + offset * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    return image


# ============================================================
# SINGLE-IMAGE WORKERS
# ============================================================
def worker_rf(image_path, result_path):
    result = run_rfdetr(image_path)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def worker_lane(image_path, result_path):
    result = {"lanes": run_curvelanes(image_path)}
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def worker_metric(image_path, detections_path, result_path, depth_scale):
    with open(detections_path, "r", encoding="utf-8") as handle:
        detections = json.load(handle)
    result = run_metric3d(image_path, detections, depth_scale, save_artifacts=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def run_worker(python_executable, worker_name, image_path, result_path,
               detections_path=None, depth_scale=DEFAULT_DEPTH_SCALE):
    command = [
        str(python_executable), str(Path(__file__).resolve()),
        "--worker", worker_name, "--image", str(image_path), "--result", str(result_path),
    ]
    if worker_name == "metric":
        if detections_path is None:
            raise ValueError("Metric3D worker requires the RF-DETR detections JSON.")
        command.extend(["--detections-file", str(detections_path), "--depth-scale", str(depth_scale)])
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"{worker_name} worker failed with exit code {completed.returncode}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def process_still_image(image_path, depth_scale=DEFAULT_DEPTH_SCALE,
                        run_lanes=True, output_path=None, announce=True):
    check_paths(require_lane=run_lanes)
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    tag = f"{image_path.stem}_{time.time_ns()}"
    rf_result_path = TMP_DIR / f"rfdetr_{tag}.json"
    lane_result_path = TMP_DIR / f"curvelanes_{tag}.json"
    metric_result_path = TMP_DIR / f"metric3d_{tag}.json"
    temp_paths = [rf_result_path, lane_result_path, metric_result_path]

    try:
        run_worker(RF_PYTHON, "rf", image_path, rf_result_path)
        detections = load_json(rf_result_path)

        lanes = []
        if run_lanes:
            run_worker(LANE_PYTHON, "lane", image_path, lane_result_path)
            lanes = load_json(lane_result_path).get("lanes", [])
        selected_lanes = select_two_lane_boundaries(lanes, image.shape)

        run_worker(
            METRIC_PYTHON, "metric", image_path, metric_result_path,
            detections_path=rf_result_path, depth_scale=depth_scale,
        )
        metric_result = load_json(metric_result_path)
        output_path = Path(output_path) if output_path else OUTPUT_DIR / f"visentra_{image_path.stem}_result.jpg"
        annotated = draw_results(
            image, detections, selected_lanes, output_path,
            distances=metric_result.get("distances", []),
            depth_scale=depth_scale,
            status_lines=["Still-image RF-DETR + Metric3D result"],
        )
        if announce:
            print(f"Objects detected: {len(detections.get('boxes', []))}; "
                  f"selected lane boundaries: {len(selected_lanes)}")
            skipped_non_vehicles = 0
            for item in metric_result.get("distances", []):
                if item.get("status") == "non_vehicle_skipped":
                    skipped_non_vehicles += 1
                    continue
                est = item.get("distance_m")
                name = item.get("class_name", "vehicle")
                conf = item.get("confidence")
                if est is None:
                    print(f"  Vehicle detection {item['detection_index']} ({name}): "
                          f"distance unavailable; {item.get('status')}")
                else:
                    conf_text = f", confidence={conf:.2f}" if conf is not None else ""
                    print(f"  Vehicle detection {item['detection_index']} ({name}{conf_text}): "
                          f"~{est:.2f} m [experimental]")
            if skipped_non_vehicles:
                print(f"Ignored {skipped_non_vehicles} non-vehicle detection(s) for distance estimation.")
            print(f"Saved annotated image: {output_path}")
            if metric_result.get("raw_depth_path"):
                print(f"Saved raw depth: {metric_result['raw_depth_path']}")
                print(f"Saved depth visualization: {metric_result['depth_visualization_path']}")
        return {
            "image_path": str(image_path),
            "detections": detections,
            "lanes": selected_lanes,
            "metric": metric_result,
            "output_path": str(output_path),
            "annotated_image": annotated,
        }
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def master(image_path, depth_scale=DEFAULT_DEPTH_SCALE, run_lanes=True):
    return process_still_image(image_path, depth_scale=depth_scale, run_lanes=run_lanes)


# ============================================================
# LOCAL TARGET SELECTION (NO IMAGE UPLOAD REQUIRED)
# ============================================================
def choose_reference_vehicle_index(label, image_path, result):
    """Show detections locally, then let the user choose the known-distance target."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for target selection: {image_path}")

    detections = result.get("detections", {})
    distance_items = result.get("metric", {}).get("distances", [])
    distance_by_index = {int(item.get("detection_index", -1)): item for item in distance_items}
    boxes = detections.get("boxes", []) or []
    names = detections.get("class_names", []) or []
    confidences = detections.get("confidences", []) or []
    valid_vehicle_indices = []

    preview = image.copy()
    for index, box in enumerate(boxes):
        name = names[index] if index < len(names) else "object"
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        is_vehicle = is_vehicle_class(name)
        if is_vehicle:
            valid_vehicle_indices.append(index)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(x1, preview.shape[1] - 1))
        x2 = max(0, min(x2, preview.shape[1] - 1))
        y1 = max(0, min(y1, preview.shape[0] - 1))
        y2 = max(0, min(y2, preview.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 210, 0) if is_vehicle else (130, 130, 130)
        entry = distance_by_index.get(index, {})
        estimate = entry.get("distance_m")
        if is_vehicle and estimate is not None:
            label_text = f"#{index} {name} {confidence:.2f} ~{float(estimate):.1f}m"
        elif is_vehicle:
            label_text = f"#{index} {name} {confidence:.2f} distance unavailable"
        else:
            label_text = f"#{index} {name} {confidence:.2f} [not a vehicle target]"

        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        text_y = max(22, y1 - 8)
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
        )
        bg_x2 = min(preview.shape[1] - 1, x1 + text_w + 8)
        bg_y1 = max(0, text_y - text_h - 7)
        cv2.rectangle(preview, (x1, bg_y1), (bg_x2, text_y + 4), color, -1)
        cv2.putText(preview, label_text, (x1 + 3, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)

    header = f"{label}: identify the intended target vehicle; IDs match terminal prompt"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (30, 30, 30), -1)
    cv2.putText(preview, header[:110], (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)

    max_w, max_h = 1100, 720
    scale = min(max_w / preview.shape[1], max_h / preview.shape[0], 1.0)
    display = preview
    if scale < 1.0:
        display = cv2.resize(
            preview, (int(preview.shape[1] * scale), int(preview.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    window_name = f"VISENTRA target selection - {label}"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display)
        print(f"\n[{label}] A local window shows numbered detections and distances.")
        print("Identify the SAME vehicle whose reference distance was measured.")
        print("Press any key while the image window is active; then enter its # ID in PowerShell.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        raise RuntimeError(
            "Could not open the local target-selection window. Run this command from an "
            "interactive Windows desktop session. No image upload is needed."
        ) from exc

    available = sorted(set(valid_vehicle_indices))
    if not available:
        print(f"No supported vehicle detection is available for {label}.")
        return None

    while True:
        raw = input(
            f"Enter target vehicle detection ID for {label} {available} "
            "(Enter = skip): "
        ).strip()
        if not raw:
            return None
        try:
            index = int(raw.lstrip("#"))
        except ValueError:
            print("Enter an integer detection ID, or press Enter to skip.")
            continue
        if index not in available:
            print(f"That ID is not a supported vehicle detection. Choose one of {available}.")
            continue
        return index


# ============================================================
# KNOWN-DISTANCE TEST MODE (RF-DETR FIRST, THEN METRIC3D)
# ============================================================
def test_known_distances(depth_scale=DEFAULT_DEPTH_SCALE, with_lanes=False,
                         select_targets=False):
    test_root = ROOT / r"MonoPilot\out\independent_tests"
    test_cases = [("5m", 5.0), ("10m", 10.0), ("15m", 15.0), ("20m", 20.0)]
    report_rows = []
    target_rows = []

    for label, reference_distance in test_cases:
        image_path = test_root / label / f"{label}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Known-distance image not found: {image_path}")
        print("\n" + "=" * 65)
        print(f"KNOWN-DISTANCE TEST: {label} (reference distance {reference_distance:.1f} m)")
        output_path = OUTPUT_DIR / f"visentra_{label}_rfdetr_metric3d.jpg"
        result = process_still_image(
            image_path,
            depth_scale=depth_scale,
            run_lanes=with_lanes,
            output_path=output_path,
        )
        distances = result["metric"].get("distances", [])
        vehicle_distances = [
            item for item in distances
            if item.get("status") != "non_vehicle_skipped"
        ]

        # Optional local-only target selection. The image stays on the user's PC.
        selected_index = None
        if select_targets:
            selected_index = choose_reference_vehicle_index(label, image_path, result)
            selected = next(
                (item for item in vehicle_distances
                 if int(item.get("detection_index", -1)) == selected_index),
                None,
            ) if selected_index is not None else None
            if selected is None:
                target_rows.append({
                    "image": label,
                    "reference_distance_m": reference_distance,
                    "target_detection_index": "",
                    "class_name": "",
                    "confidence": "",
                    "raw_depth_median": "",
                    "estimated_distance_m": "",
                    "error_m": "",
                    "absolute_error_m": "",
                    "status": "target_skipped_or_unavailable",
                    "output_image": str(output_path),
                })
            else:
                estimated = selected.get("distance_m")
                error = (float(estimated) - reference_distance) if estimated is not None else None
                target_rows.append({
                    "image": label,
                    "reference_distance_m": reference_distance,
                    "target_detection_index": selected_index,
                    "class_name": selected.get("class_name", "vehicle"),
                    "confidence": selected.get("confidence", ""),
                    "raw_depth_median": selected.get("raw_depth_median", ""),
                    "estimated_distance_m": estimated if estimated is not None else "",
                    "error_m": round(error, 3) if error is not None else "",
                    "absolute_error_m": round(abs(error), 3) if error is not None else "",
                    "status": selected.get("status", "unknown"),
                    "output_image": str(output_path),
                })
                if estimated is None:
                    print(f"Selected target #{selected_index}, but its distance is unavailable.")
                else:
                    print(
                        f"Selected reference target #{selected_index}: "
                        f"estimated={float(estimated):.2f} m, "
                        f"reference={reference_distance:.2f} m, "
                        f"error={float(estimated) - reference_distance:+.2f} m"
                    )

        if not vehicle_distances:
            print("No supported vehicle-class detections returned by RF-DETR for this image.")
            report_rows.append({
                "image": label,
                "reference_distance_m": reference_distance,
                "detection_index": "",
                "class_name": "",
                "confidence": "",
                "raw_depth_median": "",
                "estimated_distance_m": "",
                "valid_pixel_fraction": "",
                "capped_pixel_fraction": "",
                "status": "no_vehicle_detections",
                "output_image": str(output_path),
            })
        else:
            for item in vehicle_distances:
                report_rows.append({
                    "image": label,
                    "reference_distance_m": reference_distance,
                    "detection_index": item.get("detection_index"),
                    "class_name": item.get("class_name"),
                    "confidence": item.get("confidence"),
                    "raw_depth_median": item.get("raw_depth_median"),
                    "estimated_distance_m": item.get("distance_m"),
                    "valid_pixel_fraction": item.get("valid_pixel_fraction"),
                    "capped_pixel_fraction": item.get("capped_pixel_fraction"),
                    "status": item.get("status"),
                    "output_image": str(output_path),
                })

    report_path = OUTPUT_DIR / "visentra_rfdetr_metric3d_known_distance_report.csv"
    if report_rows:
        report_fields = [
            "image", "reference_distance_m", "detection_index", "class_name",
            "confidence", "raw_depth_median", "estimated_distance_m",
            "valid_pixel_fraction", "capped_pixel_fraction", "status", "output_image",
        ]
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=report_fields)
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n========== FOUR-IMAGE REPORT ==========")
    for row in report_rows:
        estimated = row.get("estimated_distance_m")
        if row.get("status") == "no_vehicle_detections":
            print(f"{row['image']}: no supported vehicle detection")
        elif estimated in (None, ""):
            print(f"{row['image']}: vehicle det#{row.get('detection_index')}; "
                  f"distance unavailable ({row['status']})")
        else:
            print(f"{row['image']} reference={float(row['reference_distance_m']):.1f} m | "
                  f"vehicle det#{row['detection_index']} {row['class_name']} | "
                  f"Metric3D estimate ~{float(estimated):.2f} m (experimental)")
    print("Report CSV:", report_path)
    print("Annotated images:", OUTPUT_DIR)

    if select_targets:
        selected_report_path = OUTPUT_DIR / "visentra_rfdetr_metric3d_selected_target_validation.csv"
        selected_fields = [
            "image", "reference_distance_m", "target_detection_index", "class_name",
            "confidence", "raw_depth_median", "estimated_distance_m", "error_m",
            "absolute_error_m", "status", "output_image",
        ]
        with selected_report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected_fields)
            writer.writeheader()
            writer.writerows(target_rows)
        print("Local target-specific report CSV:", selected_report_path)
        print("\n========== SELECTED TARGET SUMMARY ==========")
        scored_rows = [row for row in target_rows if row.get("absolute_error_m") not in (None, "")]
        for row in target_rows:
            if row.get("absolute_error_m") in (None, ""):
                print(f"{row['image']}: {row['status']}")
            else:
                print(
                    f"{row['image']} target=#{row['target_detection_index']} "
                    f"reference={float(row['reference_distance_m']):.1f} m | "
                    f"estimate={float(row['estimated_distance_m']):.2f} m | "
                    f"error={float(row['error_m']):+.2f} m"
                )
        if scored_rows:
            mae = sum(float(row["absolute_error_m"]) for row in scored_rows) / len(scored_rows)
            print(f"Selected targets scored: {len(scored_rows)}; MAE on these same four calibration scenes: {mae:.2f} m")
        print("Reminder: this report uses the existing provisional scale and is not independent validation.")

    print("Note: only the manually selected target detection is compared to each reference distance.")


# ============================================================
# PERSISTENT LIVE-VIDEO WORKERS
# ============================================================
def worker_stream(worker_name, depth_scale=DEFAULT_DEPTH_SCALE):
    """Read one JSON request per line; emit one JSON response per line."""
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        frame_id = None
        try:
            request = json.loads(raw_line)
            frame_id = request.get("frame_id")
            image_path = request.get("image_path")
            if not image_path:
                raise ValueError("Worker request has no image_path.")

            with redirect_stdout(sys.stderr):
                if worker_name == "rf":
                    result = run_rfdetr(image_path)
                elif worker_name == "lane":
                    result = {"lanes": run_curvelanes(image_path)}
                elif worker_name == "metric":
                    detections = request.get("detections")
                    if not isinstance(detections, dict):
                        raise ValueError("Metric3D request has no detections object.")
                    result = run_metric3d(image_path, detections, depth_scale, save_artifacts=False)
                else:
                    raise ValueError(f"Unknown worker: {worker_name}")
            print(json.dumps({"ok": True, "frame_id": frame_id, "result": result}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "frame_id": frame_id, "error": repr(exc)}), flush=True)


def start_stream_worker(python_executable, worker_name, depth_scale=DEFAULT_DEPTH_SCALE):
    worker_env = os.environ.copy()
    # Limit per-worker CPU thread pools so the three CPU models do not oversubscribe the laptop.
    worker_env["OMP_NUM_THREADS"] = "2"
    worker_env["MKL_NUM_THREADS"] = "2"
    worker_env["OPENBLAS_NUM_THREADS"] = "1"
    worker_env["VISENTRA_ORT_THREADS"] = "2"
    command = [
        str(python_executable), str(Path(__file__).resolve()),
        "--worker-stream", worker_name, "--depth-scale", str(depth_scale),
    ]
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=worker_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )


def _put_latest(work_queue, item):
    if work_queue.full():
        try:
            work_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        work_queue.put_nowait(item)
    except queue.Full:
        pass


def _stop_process(process):
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def master_live(
    camera_index=0,
    width=640,
    height=360,
    depth_scale=DEFAULT_DEPTH_SCALE,
    run_lanes=False,
    ai_interval_seconds=2.0,
    lane_every=3,
):
    """Keep camera preview live while AI analyzes throttled, frame-matched snapshots.

    CPU-heavy AI results are displayed in a second window on their exact source frame.
    The live camera preview is always the latest camera frame; stale boxes are never
    drawn over the new live frame.
    """
    check_paths(require_lane=run_lanes)
    cv2.setNumThreads(1)

    print("[VISENTRA] Optimized live mode starting.")
    print(f"[VISENTRA] Camera requested: {width}x{height}")
    print(f"[VISENTRA] AI sample interval: {ai_interval_seconds:.2f} seconds")
    print("[VISENTRA] Lane worker:", "enabled (sampled)" if run_lanes else "disabled to reduce CPU load")
    print("[VISENTRA] Two windows will be shown:")
    print("  1) Live Camera Preview: current camera image with no stale boxes")
    print("  2) Latest AI Result: detections and distances on their matching frame")

    rf_process = start_stream_worker(RF_PYTHON, "rf", depth_scale)
    metric_process = start_stream_worker(METRIC_PYTHON, "metric", depth_scale)
    lane_process = start_stream_worker(LANE_PYTHON, "lane", depth_scale) if run_lanes else None

    processes = {"rf": rf_process, "metric": metric_process}
    if lane_process is not None:
        processes["lane"] = lane_process

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        for process in processes.values():
            _stop_process(process)
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Confirm the phone is exposed "
            "as a Windows webcam, then try --camera 0, --camera 1, or --camera 2."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    queues = {
        "rf": queue.Queue(maxsize=1),
        "metric": queue.Queue(maxsize=1),
    }
    if run_lanes:
        queues["lane"] = queue.Queue(maxsize=1)

    state = {
        "rf": {
            "result": {"boxes": [], "class_ids": [], "class_names": [], "confidences": []},
            "frame_id": None, "completed_at": None, "fps": 0.0,
            "error": None, "latency_sec": None,
        },
        "metric": {
            "result": {"distances": []}, "detections": None,
            "frame": None, "frame_id": None, "completed_at": None,
            "fps": 0.0, "error": None, "latency_sec": None,
        },
        "lane": {
            "result": {"lanes": []}, "frame_id": None,
            "completed_at": None, "fps": 0.0, "error": None,
        },
    }
    lock = threading.Lock()
    stop_event = threading.Event()
    lane_cache = {}
    lane_cache_limit = 12
    rf_completed_count = [0]  # incremented only by the single RF-DETR stream thread
    threads = []

    def call_worker(name, frame, frame_id, captured_at, detections=None):
        process = processes[name]
        frame_path = TMP_DIR / f"live_{name}_{frame_id}_{time.time_ns()}.jpg"
        try:
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise RuntimeError(f"Could not write temporary frame: {frame_path}")
            if process.poll() is not None:
                raise RuntimeError(f"{name} worker stopped unexpectedly")

            request = {"frame_id": frame_id, "image_path": str(frame_path)}
            if name == "metric":
                request["detections"] = detections
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response_line = process.stdout.readline()
            if not response_line:
                raise RuntimeError(f"{name} worker closed its output stream")
            response = json.loads(response_line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "unknown worker error"))
            if response.get("frame_id") != frame_id:
                raise RuntimeError(
                    f"{name} frame mismatch: sent {frame_id}, received {response.get('frame_id')}"
                )
            completed = time.perf_counter()
            result = response.get("result", {})

            with lock:
                previous = state[name]["completed_at"]
                state[name]["fps"] = (
                    1.0 / (completed - previous)
                    if previous is not None and completed > previous else 0.0
                )
                state[name]["completed_at"] = completed
                state[name]["frame_id"] = frame_id
                state[name]["error"] = None
                if name == "rf":
                    state["rf"]["result"] = result
                    state["rf"]["latency_sec"] = completed - captured_at
                elif name == "metric":
                    state["metric"].update({
                        "result": result,
                        "detections": detections,
                        "frame": frame.copy(),
                        "latency_sec": completed - captured_at,
                    })
                else:
                    state["lane"]["result"] = result
                    lane_cache[frame_id] = result.get("lanes", [])
                    if len(lane_cache) > lane_cache_limit:
                        for old_id in sorted(lane_cache)[:-lane_cache_limit]:
                            lane_cache.pop(old_id, None)

            # RF-DETR output is the gate for both downstream workers. This preserves
            # the exact same input frame and bounding boxes for distance estimation.
            if name == "rf":
                rf_completed_count[0] += 1
                _put_latest(queues["metric"], (frame.copy(), frame_id, captured_at, result))
                if run_lanes and lane_process is not None and (rf_completed_count[0] % max(1, lane_every) == 0):
                    _put_latest(queues["lane"], (frame.copy(), frame_id, captured_at))
        except Exception as exc:
            with lock:
                state[name]["error"] = str(exc)
            print(f"[VISENTRA] {name} inference error: {exc}", file=sys.stderr)
            if process.poll() is not None:
                stop_event.set()
        finally:
            try:
                frame_path.unlink(missing_ok=True)
            except Exception:
                pass

    def stream_loop(name):
        while not stop_event.is_set():
            try:
                item = queues[name].get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if name == "metric":
                frame, frame_id, captured_at, detections = item
                call_worker(name, frame, frame_id, captured_at, detections=detections)
            else:
                frame, frame_id, captured_at = item
                call_worker(name, frame, frame_id, captured_at)

    def lane_loop():
        while not stop_event.is_set():
            try:
                item = queues["lane"].get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, frame_id, captured_at = item
            call_worker("lane", frame, frame_id, captured_at)

    rf_thread = threading.Thread(target=stream_loop, args=("rf",), daemon=True, name="VISENTRA-RF")
    metric_thread = threading.Thread(target=stream_loop, args=("metric",), daemon=True, name="VISENTRA-Metric3D")
    threads.extend([rf_thread, metric_thread])
    if run_lanes:
        threads.append(threading.Thread(target=lane_loop, daemon=True, name="VISENTRA-Lane"))
    for thread in threads:
        thread.start()

    frame_number = 0
    camera_fps = 0.0
    previous_camera_time = time.perf_counter()
    last_ai_submit = -float(ai_interval_seconds)
    last_render_key = None
    latest_annotated = None
    last_preview = None
    first_display_message = True
    print("[VISENTRA] Live preview is active. Press Q or ESC in either window to stop.")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[VISENTRA] Could not read a camera frame; stopping.")
                break
            frame_number += 1
            now = time.perf_counter()
            delta = now - previous_camera_time
            if delta > 0:
                inst = 1.0 / delta
                camera_fps = inst if camera_fps == 0 else 0.85 * camera_fps + 0.15 * inst
            previous_camera_time = now

            # Submit at a controlled interval. The preview is not blocked on inference.
            if now - last_ai_submit >= ai_interval_seconds:
                _put_latest(queues["rf"], (frame.copy(), frame_number, now))
                last_ai_submit = now

            with lock:
                rf_fps = state["rf"]["fps"]
                metric_fps = state["metric"]["fps"]
                lane_fps = state["lane"]["fps"]
                rf_error = state["rf"]["error"]
                metric_error = state["metric"]["error"]
                lane_error = state["lane"]["error"]
                metric_frame = state["metric"]["frame"].copy() if state["metric"]["frame"] is not None else None
                metric_frame_id = state["metric"]["frame_id"]
                metric_completed = state["metric"]["completed_at"]
                metric_latency = state["metric"]["latency_sec"]
                metric_detections = state["metric"]["detections"]
                metric_result = dict(state["metric"]["result"])
                matched_lanes = lane_cache.get(metric_frame_id, []) if metric_frame_id is not None else []
                has_matching_lanes = metric_frame_id in lane_cache if metric_frame_id is not None else False

            ai_age = (now - metric_completed) if metric_completed is not None else None
            preview = frame.copy()
            cv2.putText(preview, "LIVE CAMERA PREVIEW - current frame", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, f"Camera FPS: {camera_fps:.1f} | frame: {frame_number}",
                        (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            fps_text = f"AI FPS RF: {rf_fps:.2f} | Metric3D: {metric_fps:.2f}"
            if run_lanes:
                fps_text += f" | Lane: {lane_fps:.2f}"
            else:
                fps_text += " | Lane: OFF (to reduce CPU load)"
            cv2.putText(preview, fps_text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 2, cv2.LINE_AA)
            if metric_frame_id is not None and ai_age is not None:
                cv2.putText(preview,
                            f"Last AI result frame #{metric_frame_id}; result age: {ai_age:.1f}s; pipeline latency: {metric_latency or 0:.1f}s",
                            (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(preview, "Waiting for first complete RF-DETR + Metric3D result...",
                            (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (0, 255, 255), 2, cv2.LINE_AA)
            errors = []
            if rf_error:
                errors.append("RF-DETR: " + rf_error)
            if metric_error:
                errors.append("Metric3D: " + metric_error)
            if run_lanes and lane_error:
                errors.append("CurveLaneS: " + lane_error)
            if errors:
                cv2.putText(preview, "WORKER ERROR - see PowerShell", (12, 138),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            last_preview = preview
            cv2.imshow("VISENTRA - Live Camera Preview", preview)

            # Render the last finished AI result only when its ID or its lane overlay changes.
            # It is deliberately shown separately to avoid drawing old boxes on current footage.
            render_key = (metric_frame_id, has_matching_lanes)
            if metric_frame is None:
                waiting = frame.copy()
                cv2.putText(waiting, "AI result window: waiting for first complete inference...",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("VISENTRA - Latest AI Result (matched frame)", waiting)
            elif render_key != last_render_key:
                status_lines = [
                    f"AI pipeline latency: {metric_latency:.2f}s" if metric_latency is not None else "AI pipeline latency: n/a",
                    f"Result age at display: {ai_age:.2f}s" if ai_age is not None else "Result age: n/a",
                    "Matched original frame; experimental distance, not safety validated",
                ]
                if run_lanes and not has_matching_lanes:
                    status_lines.append("Matching lane result pending")
                latest_annotated = draw_results(
                    metric_frame,
                    metric_detections or {"boxes": [], "class_ids": [], "class_names": [], "confidences": []},
                    select_two_lane_boundaries(matched_lanes, metric_frame.shape),
                    distances=metric_result.get("distances", []),
                    depth_scale=depth_scale,
                    frame_number=metric_frame_id,
                    fps=metric_fps,
                    status_lines=status_lines,
                )
                cv2.imshow("VISENTRA - Latest AI Result (matched frame)", latest_annotated)
                last_render_key = render_key

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        for work_queue in queues.values():
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                try:
                    work_queue.get_nowait()
                    work_queue.put_nowait(None)
                except queue.Empty:
                    pass
        for process in processes.values():
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            _stop_process(process)
        for thread in threads:
            thread.join(timeout=1)

    if latest_annotated is not None:
        output_path = OUTPUT_DIR / "visentra_live_result.jpg"
        cv2.imwrite(str(output_path), latest_annotated)
        print(f"[VISENTRA] Last matched AI result saved: {output_path}")
    if last_preview is not None:
        preview_path = OUTPUT_DIR / "visentra_live_preview.jpg"
        cv2.imwrite(str(preview_path), last_preview)
        print(f"[VISENTRA] Last live preview saved: {preview_path}")
    print("[VISENTRA] Live test ended.")

# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VISENTRA: RF-DETR object detection + Metric3D v2 depth + CurveLaneS lanes"
    )
    parser.add_argument("--image", default=None, help="Input image path for still-image mode")
    parser.add_argument("--worker", choices=["rf", "lane", "metric"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result", default=None, help="Worker result JSON path")
    parser.add_argument("--detections-file", default=None,
                        help="RF-DETR JSON input for the Metric3D worker")
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE,
                        help="Provisional raw Metric3D depth multiplier; default 0.373572")
    parser.add_argument("--live", action="store_true", help="Run live camera inference")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (try 0, 1, 2)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--ai-interval-seconds", type=float, default=2.0,
                        help="Minimum spacing between submitted live AI frames; larger values reduce CPU load")
    parser.add_argument("--with-live-lanes", action="store_true",
                        help="Enable CurveLaneS in live mode; disabled by default for smoother CPU-only testing")
    parser.add_argument("--lane-every", type=int, default=3,
                        help="When --with-live-lanes is set, run lane inference on every Nth RF-DETR frame")
    parser.add_argument("--worker-stream", choices=["rf", "lane", "metric"], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--test-known-distances", action="store_true",
                        help="Run RF-DETR + Metric3D on the existing 5m/10m/15m/20m images")
    parser.add_argument("--select-targets", action="store_true",
                        help="Show local numbered detections and let you choose the measured target vehicle in each test image")
    parser.add_argument("--with-lanes", action="store_true",
                        help="Also run CurveLaneS during --test-known-distances")
    parser.add_argument("--skip-lanes", action="store_true",
                        help="Skip CurveLaneS for a faster still-image check")
    args = parser.parse_args()

    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("--depth-scale must be a positive finite number")

    if args.worker_stream is not None:
        worker_stream(args.worker_stream, args.depth_scale)
        return

    if args.worker is not None:
        if args.image is None:
            parser.error("--image is required in worker mode")
        if args.result is None:
            parser.error("--result is required in worker mode")
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        if args.worker == "rf":
            worker_rf(image_path, args.result)
        elif args.worker == "lane":
            worker_lane(image_path, args.result)
        else:
            if args.detections_file is None:
                parser.error("--detections-file is required for Metric3D worker mode")
            worker_metric(image_path, args.detections_file, args.result, args.depth_scale)
        return

    if args.select_targets and not args.test_known_distances:
        parser.error("--select-targets can only be used with --test-known-distances")

    if args.test_known_distances:
        test_known_distances(
            depth_scale=args.depth_scale,
            with_lanes=args.with_lanes,
            select_targets=args.select_targets,
        )
        return

    if args.live:
        if args.width <= 0 or args.height <= 0:
            parser.error("--width and --height must be positive integers")
        if not np.isfinite(args.ai_interval_seconds) or args.ai_interval_seconds <= 0:
            parser.error("--ai-interval-seconds must be a positive number")
        if args.lane_every < 1:
            parser.error("--lane-every must be at least 1")
        master_live(
            args.camera, args.width, args.height, args.depth_scale,
            run_lanes=args.with_live_lanes and not args.skip_lanes,
            ai_interval_seconds=args.ai_interval_seconds,
            lane_every=args.lane_every,
        )
        return

    if args.image is None:
        parser.error("Use --test-known-distances, --live, or --image for still-image mode")
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    master(image_path, depth_scale=args.depth_scale, run_lanes=not args.skip_lanes)


if __name__ == "__main__":
    main()
