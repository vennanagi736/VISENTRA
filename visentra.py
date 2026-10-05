import os
import sys
import cv2
import json
import subprocess
import tempfile
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# VISENTRA ROOT
# ============================================================

ROOT = Path(r"V:\visentra")


# ============================================================
# PYTHON ENVIRONMENTS
# ============================================================

RF_PYTHON = (
    ROOT /
    r"environments\rfdetr_env\Scripts\python.exe"
)

LANE_PYTHON = (
    ROOT /
    r"environments\condlanenet_env\Scripts\python.exe"
)


# ============================================================
# MODELS
# ============================================================

RF_MODEL = (
    ROOT /
    r"models\rf-detr-medium.pth"
)

LANE_MODEL = (
    ROOT /
    r"models\curvelanes_medium.pth"
)


# ============================================================
# CURVELANES
# ============================================================

COND_LANE_ROOT = (
    ROOT /
    "CondLaneNet"
)

LANE_CONFIG = (
    COND_LANE_ROOT /
    r"configs\condlanenet\curvelanes\curvelanes_medium_test.py"
)


# ============================================================
# OUTPUT / TEMP
# ============================================================

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TMP_DIR = ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True)


# ============================================================
# TEMP ENVIRONMENT
# ============================================================

os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)

tempfile.tempdir = str(TMP_DIR)


# ============================================================
# PATH CHECK
# ============================================================

def check_paths():

    paths = {
        "RF-DETR Python": RF_PYTHON,
        "CurveLaneS Python": LANE_PYTHON,
        "RF-DETR model": RF_MODEL,
        "CurveLaneS model": LANE_MODEL,
        "CurveLaneS config": LANE_CONFIG,
        "CondLaneNet": COND_LANE_ROOT,
    }

    print()
    print("========== VISENTRA PATH CHECK ==========")

    for name, path in paths.items():

        if path.exists():
            print(f"[OK] {name}: {path}")

        else:
            print(f"[ERROR] Missing {name}: {path}")
            raise FileNotFoundError(path)

    print("=========================================")
    print()


# ============================================================
# RF-DETR
# ============================================================

def run_rfdetr(image_path):

    print("[RF-DETR] Loading pretrained model...")

    from rfdetr import from_checkpoint

    model = from_checkpoint(
        str(RF_MODEL)
    )

    print("[OK] RF-DETR loaded.")

    print("[RF-DETR] Running inference...")

    detections = model.predict(
        str(image_path),
        threshold=0.40
    )

    count = len(detections.class_id)

    print(
        f"[RF-DETR] {count} detections found."
    )

    result = {
        "boxes": [],
        "class_ids": [],
        "confidences": []
    }

    for box, class_id, confidence in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):

        result["boxes"].append([
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3])
        ])

        result["class_ids"].append(
            int(class_id)
        )

        result["confidences"].append(
            float(confidence)
        )

    return result


# ============================================================
# CURVELANES MODEL LOADING
# ============================================================

def load_curvelanes():

    print("[CurveLaneS] Loading pretrained model...")

    # --------------------------------------------------------
    # CondLaneNet must be first on Python path
    # --------------------------------------------------------

    cond_root = str(
        COND_LANE_ROOT
    )

    if cond_root not in sys.path:
        sys.path.insert(
            0,
            cond_root
        )

    # --------------------------------------------------------
    # Torch
    # --------------------------------------------------------

    import torch

    # --------------------------------------------------------
    # CurveLaneS was originally written for CUDA.
    #
    # This installation is CPU-only.
    #
    # The old model explicitly references:
    #
    # torch.cuda.FloatTensor
    #
    # Make that reference point to CPU FloatTensor.
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        print(
            "[CurveLaneS] CUDA unavailable."
        )

        print(
            "[CurveLaneS] Applying CPU compatibility patch..."
        )

        torch.cuda.FloatTensor = torch.FloatTensor

        print(
            "[OK] CPU compatibility patch applied."
        )

    # --------------------------------------------------------
    # Old MMCV / MMDetection
    # --------------------------------------------------------

    import mmcv

    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    # --------------------------------------------------------
    # Load config directly.
    #
    # Config.fromfile() causes the Windows temp-file issue
    # with this old MMCV version.
    # --------------------------------------------------------

    config_namespace = {
        "__file__": str(LANE_CONFIG),
        "__name__": "__curvelanes_config__"
    }

    with open(
        LANE_CONFIG,
        "r",
        encoding="utf-8"
    ) as f:

        config_source = f.read()

    exec(
        compile(
            config_source,
            str(LANE_CONFIG),
            "exec"
        ),
        config_namespace
    )

    cfg_dict = {
        key: value
        for key, value in config_namespace.items()
        if not key.startswith("__")
    }

    cfg = mmcv.Config(
        cfg_dict
    )

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cfg.model.head.location_configs.device = "cpu"

    cfg.model.pretrained = None

    print(
        "[CurveLaneS] Building model..."
    )

    model = build_detector(
        cfg.model
    )

    print(
        "[CurveLaneS] Loading checkpoint..."
    )

    load_checkpoint(
        model,
        str(LANE_MODEL),
        map_location="cpu"
    )

    model = model.cpu()

    model.eval()

    print(
        "[OK] CurveLaneS loaded."
    )

    return model, cfg


# ============================================================
# CURVELANES PREPROCESSING
# ============================================================

def prepare_curvelanes_image(
    image_path,
    cfg
):

    from mmdet.datasets.pipelines import Compose

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    img = cv2.imread(
        str(image_path)
    )

    if img is None:

        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    original_shape = img.shape

    print(
        "[CurveLaneS] Original image: "
        f"{original_shape[1]}x{original_shape[0]}"
    )

    # --------------------------------------------------------
    # Arbitrary image input.
    #
    # We intentionally bypass CurvelanesDataset because
    # that dataset only accepts its native image sizes.
    # --------------------------------------------------------

    results = {

        "filename": str(
            image_path
        ),

        "sub_img_name": Path(
            image_path
        ).name,

        "img": img,

        "gt_points": [],

        "id_classes": [],

        "id_instances": [],

        "img_shape": original_shape,

        "ori_shape": original_shape,

        "crop_offset": [0, 0],

        "crop_shape": original_shape,
    }

    # --------------------------------------------------------
    # Exact validation pipeline from CurveLaneS config
    # --------------------------------------------------------

    pipeline = Compose(
        cfg.val_pipeline
    )

    data = pipeline(
        results
    )

    if data is None:

        raise RuntimeError(
            "CurveLaneS preprocessing returned None."
        )

    return data, original_shape


# ============================================================
# EXTRACT DATA CONTAINER
# ============================================================

def unwrap_data_container(
    value
):

    """
    Extract the actual object from an old MMCV DataContainer.

    Handles the different structures used by MMCV 0.5.x.
    """

    if not hasattr(
        value,
        "data"
    ):

        return value

    value = value.data

    # --------------------------------------------------------
    # Repeatedly unwrap single-element list/tuple containers.
    #
    # Do not unwrap dictionaries.
    # --------------------------------------------------------

    while isinstance(
        value,
        (list, tuple)
    ) and len(value) == 1:

        value = value[0]

    return value


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_inputs(
    data
):

    import torch

    # ========================================================
    # IMAGE
    # ========================================================

    img_container = data[
        "img"
    ]

    img = unwrap_data_container(
        img_container
    )

    # --------------------------------------------------------
    # Convert image to Tensor
    # --------------------------------------------------------

    if not isinstance(
        img,
        torch.Tensor
    ):

        img = torch.tensor(
            img
        )

    # --------------------------------------------------------
    # Ensure float
    # --------------------------------------------------------

    img = img.float()

    # --------------------------------------------------------
    # Ensure BCHW
    # --------------------------------------------------------

    if img.dim() == 3:

        img = img.unsqueeze(
            0
        )

    elif img.dim() != 4:

        raise RuntimeError(
            "Unexpected CurveLaneS image tensor "
            f"shape: {tuple(img.shape)}"
        )

    img = img.cpu()

    # ========================================================
    # METADATA
    # ========================================================

    meta_container = data[
        "img_metas"
    ]

    raw_meta = meta_container.data

    # --------------------------------------------------------
    # Print structure once for diagnosis.
    # --------------------------------------------------------

    print(
        "[CurveLaneS] img_metas container type: "
        f"{type(meta_container).__name__}"
    )

    print(
        "[CurveLaneS] img_metas data type: "
        f"{type(raw_meta).__name__}"
    )

    # --------------------------------------------------------
    # Extract metadata robustly.
    # --------------------------------------------------------

    meta = raw_meta

    while isinstance(
        meta,
        (list, tuple)
    ) and len(meta) == 1:

        meta = meta[0]

    # --------------------------------------------------------
    # If the result is still a list/tuple containing the
    # metadata dictionary, take the first dictionary.
    # --------------------------------------------------------

    if isinstance(
        meta,
        (list, tuple)
    ):

        if len(meta) == 0:

            raise RuntimeError(
                "CurveLaneS img_metas is empty."
            )

        if isinstance(
            meta[0],
            dict
        ):

            meta = meta[0]

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not isinstance(
        meta,
        dict
    ):

        raise RuntimeError(
            "Could not extract CurveLaneS metadata. "
            f"Final type: {type(meta).__name__}"
        )

    # --------------------------------------------------------
    # The old detector expects a list of metadata dictionaries.
    # --------------------------------------------------------

    img_metas = [
        meta
    ]

    return img, img_metas, meta


# ============================================================
# CURVELANES INFERENCE
# ============================================================

def run_curvelanes(
    image_path
):

    # --------------------------------------------------------
    # CondLaneNet must be available before importing tools.
    # --------------------------------------------------------

    cond_root = str(
        COND_LANE_ROOT
    )

    if cond_root not in sys.path:

        sys.path.insert(
            0,
            cond_root
        )

    # --------------------------------------------------------
    # Post processor
    # --------------------------------------------------------

    from tools.condlanenet.post_process import (
        CurvelanesPostProcessor
    )

    print(
        "[CurveLaneS] Preparing image..."
    )

    model, cfg = load_curvelanes()

    data, original_shape = (
        prepare_curvelanes_image(
            image_path,
            cfg
        )
    )

    # --------------------------------------------------------
    # Convert MMCV DataContainers into actual model inputs.
    # --------------------------------------------------------

    img, img_metas, meta = (
        prepare_model_inputs(
            data
        )
    )

    print(
        "[CurveLaneS] Preprocessed image: "
        f"{img.shape[-1]}x"
        f"{img.shape[-2]}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print(
        "[CurveLaneS] Running inference..."
    )

    import torch

    with torch.no_grad():

        seeds, hm = model(
            img=img,
            img_metas=img_metas,
            return_loss=False,
            rescale=False,
            thr=0.3
        )

    print(
        "[CurveLaneS] Inference complete."
    )

    # --------------------------------------------------------
    # Post processor
    # --------------------------------------------------------

    post_processor = CurvelanesPostProcessor(
        mask_size=cfg.mask_size,
        hm_thr=0.3
    )

    # --------------------------------------------------------
    # Downscale
    # --------------------------------------------------------

    downscale = meta[
        "down_scale"
    ]

    # --------------------------------------------------------
    # Post-process
    # --------------------------------------------------------

    lanes, seeds = post_processor(
        seeds,
        downscale
    )

    raw_count = (
        len(lanes)
        if lanes is not None
        else 0
    )

    print(
        f"[CurveLaneS] Raw lanes detected: "
        f"{raw_count}"
    )

    # ========================================================
    # COORDINATE CONVERSION
    # ========================================================

    processed_shape = meta[
        "img_shape"
    ]

    h_processed = (
        processed_shape[0]
    )

    w_processed = (
        processed_shape[1]
    )

    h_original = (
        original_shape[0]
    )

    w_original = (
        original_shape[1]
    )

    ratio_x = (
        w_original /
        w_processed
    )

    ratio_y = (
        h_original /
        h_processed
    )

    final_lanes = []

    if lanes is not None:

        for lane in lanes:

            points = []

            for point in lane["points"]:

                x = (
                    float(point[0]) *
                    ratio_x
                )

                y = (
                    float(point[1]) *
                    ratio_y
                )

                # ------------------------------------------------
                # Clamp coordinates
                # ------------------------------------------------

                x = max(
                    0.0,
                    min(
                        x,
                        w_original - 1
                    )
                )

                y = max(
                    0.0,
                    min(
                        y,
                        h_original - 1
                    )
                )

                points.append([
                    int(round(x)),
                    int(round(y))
                ])

            if len(points) > 1:

                final_lanes.append(
                    points
                )

    print(
        f"[CurveLaneS] Final lanes: "
        f"{len(final_lanes)}"
    )

    return final_lanes

# ============================================================
# DRAW FINAL RESULT
# ============================================================

def draw_results(
    image_path,
    detections,
    lanes,
    output_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    # ========================================================
    # COLORS
    # OpenCV uses BGR
    # ========================================================

    DARK_GREEN = (0, 140, 0)      # Vehicle boxes
    DARK_BLUE = (180, 0, 0)       # Lane lines

    # ========================================================
    # RF-DETR VEHICLES
    # ========================================================

    boxes = detections[
        "boxes"
    ]

    class_ids = detections[
        "class_ids"
    ]

    confidences = detections[
        "confidences"
    ]

    for box, class_id, confidence in zip(
        boxes,
        class_ids,
        confidences
    ):

        x1, y1, x2, y2 = [
            int(round(value))
            for value in box
        ]

        # ----------------------------------------------------
        # Vehicle bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            DARK_GREEN,
            2
        )

        # ----------------------------------------------------
        # Vehicle label
        # ----------------------------------------------------

        label = (
            f"ID {class_id} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(20, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            DARK_GREEN,
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # CURVELANES
    # ========================================================

    for lane_index, lane in enumerate(
        lanes
    ):

        if len(lane) < 2:
            continue

        points = np.array(
            lane,
            dtype=np.int32
        ).reshape(
            (-1, 1, 2)
        )

        # ----------------------------------------------------
        # Lane curve
        # ----------------------------------------------------

        cv2.polylines(
            image,
            [points],
            False,
            DARK_BLUE,
            5,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Lane label
        # ----------------------------------------------------

        x, y = lane[0]

        cv2.putText(
            image,
            f"Lane {lane_index + 1}",
            (
                x,
                max(
                    25,
                    y
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            DARK_BLUE,
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # SAVE
    # ========================================================

    cv2.imwrite(
        str(output_path),
        image
    )

    print()
    print(
        f"[SAVED] {output_path}"
    )# ============================================================
# RF-DETR WORKER
# ============================================================

def worker_rf(
    image_path,
    result_path
):

    print()
    print(
        "========== RF-DETR WORKER =========="
    )

    result = run_rfdetr(
        image_path
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"[RF-DETR] Results saved: "
        f"{result_path}"
    )

    print(
        "[RF-DETR] Worker complete."
    )


# ============================================================
# CURVELANES WORKER
# ============================================================

def worker_lane(
    image_path,
    result_path
):

    print()
    print(
        "========== CURVELANES WORKER =========="
    )

    lanes = run_curvelanes(
        image_path
    )

    result = {
        "lanes": lanes
    }

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"[CurveLaneS] Results saved: "
        f"{result_path}"
    )

    print(
        "[CurveLaneS] Worker complete."
    )


# ============================================================
# RUN SUBPROCESS
# ============================================================

def run_worker(
    python_executable,
    worker_name,
    image_path,
    result_path
):

    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--worker",
        worker_name,
        "--image",
        str(image_path),
        "--result",
        str(result_path),
    ]

    print()
    print(
        "[VISENTRA] Starting "
        f"{worker_name} worker..."
    )

    print(
        "[VISENTRA] Python: "
        f"{python_executable}"
    )

    process = subprocess.run(
        command,
        cwd=str(ROOT)
    )

    if process.returncode != 0:

        raise RuntimeError(
            f"{worker_name} worker failed "
            f"with exit code "
            f"{process.returncode}"
        )


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# MASTER
# ============================================================

def master(
    image_path
):

    print()
    print(
        "=============================================="
    )
    print(
        "              VISENTRA"
    )
    print(
        "     RF-DETR + CurveLaneS Pipeline"
    )
    print(
        "=============================================="
    )
    print()

    check_paths()

    print(
        f"[INPUT] {image_path}"
    )

    # ========================================================
    # RESULT FILES
    # ========================================================

    rf_result_path = (
        TMP_DIR /
        "rfdetr_results.json"
    )

    lane_result_path = (
        TMP_DIR /
        "curvelanes_results.json"
    )

    # ========================================================
    # REMOVE OLD RESULTS
    # ========================================================

    if rf_result_path.exists():
        rf_result_path.unlink()

    if lane_result_path.exists():
        lane_result_path.unlink()

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 1: RF-DETR VEHICLE DETECTION"
    )
    print(
        "=============================================="
    )

    run_worker(
        RF_PYTHON,
        "rf",
        image_path,
        rf_result_path
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 2: CURVELANES LANE DETECTION"
    )
    print(
        "=============================================="
    )

    run_worker(
        LANE_PYTHON,
        "lane",
        image_path,
        lane_result_path
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "STEP 3: COMBINING RESULTS"
    )
    print(
        "=============================================="
    )

    detections = load_json(
        rf_result_path
    )

    lane_data = load_json(
        lane_result_path
    )

    lanes = lane_data[
        "lanes"
    ]

    # ========================================================
    # OUTPUT
    # ========================================================

    output_path = (
        OUTPUT_DIR /
        "visentra_result.jpg"
    )

    draw_results(
        image_path,
        detections,
        lanes,
        output_path
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "           VISENTRA COMPLETE"
    )
    print(
        "=============================================="
    )

    print(
        f"Vehicles : "
        f"{len(detections['boxes'])}"
    )

    print(
        f"Lanes    : "
        f"{len(lanes)}"
    )

    print(
        f"Output   : "
        f"{output_path}"
    )

    print(
        "=============================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "VISENTRA - "
            "RF-DETR + CurveLaneS"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "--worker",
        choices=[
            "rf",
            "lane"
        ],
        default=None
    )

    parser.add_argument(
        "--result",
        default=None,
        help="Worker result JSON path"
    )

    args = parser.parse_args()

    # ========================================================
    # WORKER MODE
    # ========================================================

    if args.worker is not None:

        if args.result is None:

            raise ValueError(
                "--result is required "
                "in worker mode."
            )

        image_path = Path(
            args.image
        )

        result_path = Path(
            args.result
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Input image not found: "
                f"{image_path}"
            )

        if args.worker == "rf":

            worker_rf(
                image_path,
                result_path
            )

        elif args.worker == "lane":

            worker_lane(
                image_path,
                result_path
            )

        return

    # ========================================================
    # MASTER MODE
    # ========================================================

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Input image not found: "
            f"{image_path}"
        )

    master(
        image_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()