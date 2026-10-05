# VISENTRA

## Vehicle and Lane Detection Pipeline

VISENTRA is a computer vision pipeline that combines **RF-DETR** for vehicle detection with **CurveLaneS / CondLaneNet** for lane detection.

The system runs the two pretrained models in separate Python environments because they require different dependency versions. Their outputs are then combined into a single annotated road image.

---

## Pipeline

```text
                    VISENTRA
                       |
                       v
                 Input Road Image
                       |
              +--------+--------+
              |                 |
              v                 v
           RF-DETR          CurveLaneS
       Vehicle Detection    Lane Detection
              |                 |
              v                 v
       Vehicle Bounding      Lane Curves
            Boxes
              |                 |
              +--------+--------+
                       |
                       v
                 Result Fusion
                       |
                       v
             Final Visualization
                       |
              +--------+--------+
              |                 |
              v                 v
          Dark Green         Dark Blue
        Vehicle Boxes        Lane Curves