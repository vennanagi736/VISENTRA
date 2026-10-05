import os
import cv2
import numpy as np
import torch

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

from tools.condlanenet.post_process import CurvelanesPostProcessor


# ============================================================
# PATHS
# ============================================================

CONFIG = r"configs\condlanenet\curvelanes\curvelanes_medium_cpu.py"
CHECKPOINT = r"curvelanes_medium.pth"

INPUT_DIR = r"V:\visentra\CLRNet\test_images"
OUTPUT_DIR = r"V:\visentra\CLRNet\test_images\condlane_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEST IMAGES
# ============================================================

image_files = [
    "lane.jpg",
    "lane4.jpg",
    "lane5.jpg",
    "lane6.jpg",
    "lane7.jpg",
    "lane8.jpg",
    "lane9.jpg",
    "lane10.jpg",
    "lane11.jpg",
"all_lane_images.jpg",
]


# ============================================================
# LOAD CONFIG
# ============================================================

print("Loading config...")

cfg_namespace = {}

with open(CONFIG, "r", encoding="utf-8") as f:
    config_code = f.read()

exec(
    compile(config_code, CONFIG, "exec"),
    cfg_namespace
)

model_cfg = cfg_namespace["model"]
mask_size = cfg_namespace["mask_size"]

model_cfg["pretrained"] = None


# ============================================================
# BUILD MODEL
# ============================================================

print("Building CondLaneNet model...")

model = build_detector(model_cfg)

print("Model built successfully.")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

load_checkpoint(
    model,
    CHECKPOINT,
    map_location="cpu"
)

print("Checkpoint loaded successfully.")


# ============================================================
# CPU MODE
# ============================================================

model = model.cpu()
model.eval()

print("Using CPU.")


# ============================================================
# POST PROCESSOR
# ============================================================

post_processor = CurvelanesPostProcessor(
    mask_size=mask_size,
    hm_thr=0.5
)


# ============================================================
# NORMALIZATION
# ============================================================

mean = np.array(
    [75.3, 76.6, 77.6],
    dtype=np.float32
)

std = np.array(
    [50.5, 53.8, 54.3],
    dtype=np.float32
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("COND LANENET - VISENTRA IMAGE TEST")
print("=" * 70)
print("Images:", len(image_files))
print("Input :", INPUT_DIR)
print("Output:", OUTPUT_DIR)
print("=" * 70)
print()


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, name in enumerate(image_files, start=1):

    input_path = os.path.join(
        INPUT_DIR,
        name
    )

    print("=" * 70)
    print("Image", index, "/", len(image_files))
    print("Processing:", name)
    print("=" * 70)

    if not os.path.exists(input_path):

        print("SKIP - file not found:")
        print(input_path)
        print()

        continue


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    original = cv2.imread(input_path)

    if original is None:

        print("ERROR - could not read image")
        print(input_path)
        print()

        continue


    original_h, original_w = original.shape[:2]

    print(
        "Original size:",
        original_w,
        "x",
        original_h
    )


    # --------------------------------------------------------
    # RESIZE TO CONDLANENET INPUT
    # --------------------------------------------------------

    img = cv2.resize(
        original,
        (800, 320)
    )

    img = img.astype(
        np.float32
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    img = (
        img - mean
    ) / std


    # --------------------------------------------------------
    # HWC -> CHW -> BATCH
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        img.transpose(2, 0, 1)
    ).unsqueeze(0)


    tensor = tensor.cpu()


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print("Running inference...")

    with torch.no_grad():

        seeds, hm = model.test_inference(
            tensor
        )


    # --------------------------------------------------------
    # POST PROCESS
    # --------------------------------------------------------

    lanes, seeds = post_processor(
        seeds,
        8
    )


    # --------------------------------------------------------
    # DRAW RESULTS
    # --------------------------------------------------------

    result = original.copy()

    lane_count = 0


    if lanes is not None:

        for lane in lanes:

            points = lane.get(
                "points",
                []
            )

            if len(points) < 2:
                continue


            lane_count += 1

            pts = []


            for x, y in points:

                x = float(x)

                y = float(y)


                # Convert 800x320 coordinates
                # back to original image size

                x = (
                    x *
                    original_w /
                    800.0
                )

                y = (
                    y *
                    original_h /
                    320.0
                )


                pts.append(
                    (
                        int(x),
                        int(y)
                    )
                )


            # ------------------------------------------------
            # DRAW LANE
            # ------------------------------------------------

            for p1, p2 in zip(
                pts[:-1],
                pts[1:]
            ):

                cv2.line(
                    result,
                    p1,
                    p2,
                    (0, 255, 255),
                    4
                )


    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_name = (
        "condlane_" +
        name
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    cv2.imwrite(
        output_path,
        result
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("Detected lanes:", lane_count)
    print("Saved:", output_path)
    print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("COND LANENET TEST COMPLETE")
print("=" * 70)
print()
print("Results folder:")
print(OUTPUT_DIR)
print()