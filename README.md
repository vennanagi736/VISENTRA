

# VISENTRA
## Vision-Based Vehicle, Lane, Distance, and Trajectory-Awareness Research Pipeline

VISENTRA is a computer vision research project exploring how a road-facing camera can be used to detect vehicles, track them across frames, estimate lane boundaries, investigate approximate vehicle distances, and predict short-term vehicle motion.

The project combines **RF-DETR** for vehicle detection, **ByteTrack** for multi-frame tracking, **CurveLaneS / CondLaneNet** for lane detection, **Metric3D** for experimental monocular depth estimation, and a **Kalman-filter-based trajectory predictor** for short-term motion forecasting.

Additional depth-estimation and trajectory-prediction approaches are investigated separately. These include the MonoPilot depth-processing workstream and HiVT experiments, which must not be confused with components confirmed as integrated into the main pipeline.

VISENTRA investigates camera-only road awareness using a phone camera or compatible webcam, without requiring radar or lidar as core inputs.

**Project status:** Research and prototyping. Detection, tracking, lane estimation, depth estimation, trajectory prediction, and virtual-corridor logic have different levels of validation. The complete system is not safety-certified.

---

## 1. Why VISENTRA?

Vehicles need awareness of nearby traffic, road boundaries, and the relative positions of surrounding vehicles. Dedicated sensors can provide additional measurements, but they also introduce hardware and integration requirements.

VISENTRA investigates how much useful road information can be obtained from camera imagery and temporal computer vision.

The main objectives are:

- **Vehicle detection:** Identify vehicles visible in camera frames.
- **Vehicle tracking:** Associate detections across frames and maintain track identities.
- **Lane detection:** Estimate lane curves and visualize the apparent road corridor.
- **Distance estimation:** Investigate approximate vehicle range using monocular depth estimation.
- **Relative-position analysis:** Examine whether vehicles appear to the left, right, or center of the image.
- **Trajectory prediction:** Experiment with forecasting short-term vehicle motion from tracked observations.
- **Virtual corridor:** Investigate vehicle positions relative to a configurable image-space region.
- **Event logic:** Explore timed responses to persistent vehicle observations and multiple-object scenarios.
- **Performance evaluation:** Measure inference speed, processing throughput, skipped frames, and latency.

The long-term objective is to develop a low-cost, camera-based road-awareness prototype while measuring its limitations under repeatable test conditions.

## 2. System Architecture

VISENTRA contains related but distinct perception and experimentation workstreams.

### A. Main vehicle and lane perception pipeline

```text
                   Road Camera Frame
                           |
                           v
                        RF-DETR
                    Vehicle Detection
                           |
                           v
                 Vehicle Bounding Boxes
                           |
                           v
                       ByteTrack
                 Multi-Frame Tracking
                           |
                           v
                  Tracked Vehicle IDs
                           |
              +------------+------------+
              |            |            |
              v            v            v
        Vehicle State   CurveLaneS   Frame-Order
         and History    / CondLaneNet  Safeguard
              |          Lane Curves       |
              v               |            |
       Trajectory Predictor   |            |
       (Kalman-based)         |            |
              |               |            |
              +---------------+------------+
                              |
                              v
                   Visualization and
                  Experimental Analysis
```

The tracking frame-order safeguard prevents the tracking loop from processing a frame whose ID is not newer than the latest processed frame. It is a pipeline reliability measure, not an additional AI model.

### B. Monocular depth-estimation workstream

```text
                    Camera Image
                         |
                         v
                 Vehicle Detection
                         |
                         v
                Vehicle Image Region
                         |
                         v
              Metric Depth Estimation
                         |
                         v
               Depth and Geometry
                    Analysis
                         |
                         v
              Approximate Range Estimate
                         |
                         v
               Ground-Truth Evaluation
```

The main pipeline uses Metric3D for experimental depth estimation. MonoPilot also contains separate depth-model experiments. Their outputs and validation results should be documented independently unless a specific implementation connects them.

### C. Trajectory-prediction workstream

```text
                  Tracked Vehicle
                         |
                         v
               Timestamped Observations
                         |
                         v
            Kalman-Based State Estimation
                         |
                         v
                Short-Term Forecast
                         |
                         v
              Prediction Error Evaluation
```

Trajectory prediction uses past observations to estimate possible future image-space positions. It does not independently establish actual road-relative velocity, collision risk, or a safe maneuver.

Not every component runs simultaneously, and the presence of a model or script in the repository does not establish successful integration.

## 3. Vehicle Detection — RF-DETR

RF-DETR detects vehicles and other supported object classes in road images and video frames.

Its outputs include bounding boxes, class labels, and confidence scores.

### Role in VISENTRA

Detection provides the object locations required for tracking and subsequent analysis. The bounding boxes are passed to ByteTrack to associate detections across frames.

### Current implementation

- RF-DETR model integration in the main pipeline.
- A separate Python environment for dependency compatibility.
- Worker-based execution.
- Vehicle bounding-box visualization.
- Experimental live-video implementations.
- Diagnostic and benchmarking scripts.

Detection performance depends on image resolution, lighting, occlusion, camera position, and inference throughput. Accuracy and speed must be evaluated on representative test data.

## 4. Vehicle Tracking — ByteTrack

ByteTrack associates detections across successive frames and maintains track identities.

### Why tracking is needed

A detector processes individual frames, while tracking attempts to associate observations of the same vehicle over time.

Tracking supports:

- Maintaining vehicle track IDs.
- Recording changes in image-space position.
- Supplying observations to trajectory-prediction experiments.
- Supporting virtual-corridor and event-processing logic.
- Evaluating track continuity and recovery after missed detections.

### Tracking frame-order safeguard

The main `visentra.py` tracking loop includes a frame-order safeguard. It records the latest processed frame ID and skips incoming tracking items whose frame IDs are less than or equal to that value.

This is intended to prevent older or duplicate frames from being processed after newer frames.

The safeguard does not increase model inference speed and does not, by itself, prove improved tracking accuracy. Its effect should be evaluated using logs and repeatable video tests.

### Limitations

- **Missed detections:** Missing detections can interrupt a track.
- **ID switches:** Nearby vehicles may be assigned incorrect or changing identities.
- **Occlusion:** A partially or fully hidden vehicle may lose its track.
- **Camera movement:** Camera motion can affect apparent object displacement.
- **Low processing frequency:** Long gaps between processed observations can weaken temporal consistency.
- **Frame ordering:** Out-of-order or duplicate observations can interfere with temporal processing if not handled correctly.

ByteTrack alone does not establish accurate physical distance, relative speed, or collision risk.

## 5. Lane Detection — CurveLaneS / CondLaneNet

CurveLaneS / CondLaneNet estimates lane curves from road images.

The predicted curves can be overlaid on frames to visualize the apparent driving corridor.

### Current implementation

- CondLaneNet model integration.
- CurveLaneS configuration and checkpoint loading.
- A separate Python environment for compatibility.
- Lane inference and curve visualization.
- Experimental use alongside vehicle detections.

Lane estimates can be affected by faded markings, shadows, glare, road curvature, occlusion, lighting, and camera angle. The predicted curves should not be treated as exact road boundaries without additional validation.

## 6. Trajectory Prediction — Kalman-Based Predictor

VISENTRA includes the experimental trajectory predictor implemented in:

`visentra_trajectory_predictor.py`

The predictor uses a Kalman-filter-based, constant-acceleration approach to estimate short-term future vehicle positions from tracked observations.

### Purpose

Detection provides an object's current image-space position. Tracking supplies observations associated with a vehicle over time. The trajectory predictor uses this temporal information to estimate how the observed position might evolve over a short forecast horizon.

Potential uses include:

- Estimating short-term image-space motion.
- Visualizing predicted vehicle positions.
- Comparing predicted positions with later observations.
- Investigating the stability of motion forecasts.
- Supporting future temporal-analysis experiments.

### Current experimental constraints

The current implementation uses a bounded forecast horizon of up to approximately two seconds, with limits on estimated velocity and acceleration and damping applied to acceleration.

These are stabilization measures for the experimental predictor, not evidence of accurate real-world trajectory forecasting.

Prediction quality depends on track continuity, timestamps, observation frequency, camera movement, and tracking errors. A constant-acceleration model may not capture abrupt braking, steering, lane changes, or complex interactions between vehicles.

The predictor estimates motion in image coordinates unless additional calibrated geometry is applied. Its output must not be interpreted automatically as physical velocity, road-relative trajectory, or collision probability.

### Evaluation requirements

Future evaluation should compare predictions against later observed positions using the same recorded clips and a consistent evaluation method.

Useful measurements include:

- Mean and median position error.
- Prediction error at different forecast horizons.
- Number of valid predictions and evaluated observations.
- Performance when tracks contain missing observations.
- Sensitivity to timestamp gaps and camera motion.

Preliminary experiments are not a substitute for a larger, repeatable evaluation set.

## 7. Separate Python Environments

The models have different dependency requirements, so VISENTRA uses separate Python environments rather than forcing all dependencies into one installation.

| Component | Environment or location | Purpose |
|---|---|---|
| RF-DETR | `environments/rfdetr_env/` | Vehicle detection |
| ByteTrack | Main pipeline environment | Vehicle tracking |
| CurveLaneS / CondLaneNet | `environments/condlanenet_env/` | Lane detection |
| Metric3D | `MonoPilot/.venv/` | Experimental depth estimation |
| Trajectory predictor | Main pipeline code | Short-term motion forecasting |
| HiVT experiments | Separate experimental setup | Evaluation of an alternative trajectory-prediction approach |

These paths describe the current Windows development workspace. Other computers may require different paths, dependencies, or configuration.

## 8. MonoPilot — Monocular Depth and Distance Estimation

MonoPilot is a separate experimental workstream for monocular depth and distance estimation.

The goal is to investigate whether a single camera can provide useful approximate distance estimates for detected vehicles.

### Model and processing experiments

The repository has contained experiments involving:

- **Metric3D:** Monocular depth estimation used by the current VISENTRA depth worker.
- **Depth Anything V2:** Separate depth-model source and experimentation.
- **YOLO11n ONNX:** Vehicle-detection experiments within the MonoPilot workflow.
- **Geometric processing:** Analysis of predicted depth, image coordinates, and vehicle regions.
- **Calibration experiments:** Parameters for investigating range-estimation errors.
- **Diagnostic and benchmark scripts:** Examination of model outputs, runtime, and estimated ranges.

These components belong to distinct experimental configurations. They should not all be described as one simultaneously active model chain unless the specific tested configuration confirms that integration.

### Why combine detection and depth?

A detector identifies an object and its approximate image region. A depth model estimates depth information for image pixels. Combining them can support experiments that estimate the range of a detected vehicle.

However, camera-axis depth is not necessarily the same as the straight-line distance from the camera to a target. Bounding-box placement, selected pixels, occlusion, camera calibration, and model error can all affect the estimate.

## 9. Known-Distance Validation

A recorded MonoPilot experiment used four still images with approximate reference distances.

| Test image | Reference distance | Estimated distance | Absolute error |
|---|---:|---:|---:|
| `5m.png` | 5 m | 7.972 m | 2.972 m |
| `10m.png` | 10 m | 13.088 m | 3.088 m |
| `15m.png` | 15 m | 16.356 m | 1.356 m |
| `20m.png` | 20 m | 22.090 m | 2.090 m |
| **Mean absolute error** | — | — | **2.377 m** |

Mean absolute error (MAE) is calculated as:

\[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}
\left|d_i^{\mathrm{predicted}}-d_i^{\mathrm{reference}}\right|
\]

For the four recorded observations:

\[
\mathrm{MAE}=\frac{2.972+3.088+1.356+2.090}{4}
\approx 2.377\text{ m}
\]

### Interpretation

- Estimated distance increased with the reference distance across these four observations.
- All four estimates were greater than their reference distances.
- The mean absolute error was approximately 2.38 metres.
- The 15-metre image had the lowest absolute error in this set.

This is a small diagnostic set, not a statistically representative benchmark. It does not establish general distance accuracy or safety for vehicle operation.

Further work should include more distances, vehicles, camera positions, lighting conditions, and repeated measurements. Calibration settings should remain consistent within a test series unless a change is documented.

## 10. CPU Performance and Recorded-Video Testing

The current development machine uses Windows 11 and an Intel Core i5-1335U CPU without a dedicated GPU.

Earlier MonoPilot outdoor CPU tests ran at approximately 1.3–1.5 FPS. These results refer to that depth-processing experiment, not the complete main pipeline.

Recorded-video tests of the main pipeline have also shown substantial inference latency.

One recorded-video baseline produced the following approximate measurements:

| Pipeline component or metric | Observed result |
|---|---:|
| RF-DETR inference | 0.96 FPS |
| CurveLaneS inference | 0.94 FPS |
| Metric3D inference | 0.56 FPS |
| Capture/display throughput | 24.98 FPS |
| Capture-to-display latency | Approximately 1.10 seconds |

These are measurements from a particular test session, not universal performance figures. They should not be directly compared with results from different videos, resolutions, hardware, or configurations without accounting for those differences.

### Performance goals

The longer-term goal is to investigate 30–60 FPS processing. This target has not been demonstrated for the complete VISENTRA pipeline.

Performance reports should distinguish:

- Camera capture FPS.
- Detection inference time and FPS.
- Tracking processing time.
- Lane-detection inference time and FPS.
- Depth-estimation inference time and FPS.
- Trajectory-prediction processing time.
- End-to-end throughput.
- Frame-to-display latency.
- Dropped, skipped, and out-of-order frames.

A window that displays frames successfully does not establish that all models are processing those frames at the same rate.

## 11. Live-Processing Variants

The repository contains experimental implementations for investigating different processing approaches.

| File | Purpose |
|---|---|
| `visentra.py` | Main application and model orchestration |
| `visentra_live_accurate_low_latency.py` | Experimental live-processing implementation |
| `visentra_live_optimized.py` | Experimental live-processing optimization |
| `visentra_virtual_corridor_events.py` | Virtual-corridor and event-processing experiments |
| `visentra_trajectory_predictor.py` | Kalman-based trajectory prediction |
| `evaluate_visentra_trajectories.py` | Experimental trajectory-evaluation utility, if present in the checkout |

These variants should be tested independently. Their accuracy and performance must not be assumed equivalent.

The intended live workflow is to acquire camera frames, run detection, track vehicles, estimate lane curves, process available depth outputs, update trajectory forecasts, and display the results. Camera connectivity and throughput depend on the device, resolution, and capture method.

## 12. Virtual Corridor and Event Logic

`visentra_virtual_corridor_events.py` explores the use of a configurable image-space corridor to analyze tracked vehicle positions.

### Intended experiments

- Determine whether a tracked vehicle appears left, right, or near the center of the camera view.
- Investigate whether the vehicle appears inside or near a configured corridor.
- Analyze observations over time using track identities.
- Experiment with timed events and multiple-vehicle scenarios.
- Investigate the effects of configurable distance thresholds when a valid distance estimate is available.

### Limitations

Image-space position does not directly establish a vehicle's physical position relative to the host vehicle. A virtual corridor is not a calibrated three-dimensional safety zone.

Relative speed and collision risk cannot be established reliably from a single image. Tracking and trajectory forecasts alone do not establish those quantities either. Camera motion, calibration, missing observations, ID switches, and model errors must be considered.

All corridor and event behavior remains experimental.

## 13. HiVT and Other Candidate Models

HiVT has been explored in a separate experimental setup, including a synthetic smoke test. It is **not confirmed as integrated into the main VISENTRA pipeline**.

HiVT is a trajectory-prediction approach that can be evaluated as an alternative to the current Kalman-based predictor. A successful import or synthetic test does not establish successful integration with the real camera pipeline or accurate predictions on road video.

MobileNetV3 and DeepSORT remain candidate approaches rather than confirmed active components of the main pipeline.

- **HiVT:** Candidate for trajectory-prediction evaluation.
- **DeepSORT:** Candidate alternative to ByteTrack, not an additional tracker running alongside it.
- **MobileNetV3:** Candidate for secondary classification or feature extraction, subject to evidence that it adds value beyond RF-DETR.

Any candidate should be compared against the current implementation using the same or equivalent data, documented metrics, and reproducible configurations before it is adopted.

## 14. Repository Structure

The following paths describe principal files and folders in the development workspace. Not all local files are necessarily tracked by Git.

| Path | Purpose |
|---|---|
| `visentra.py` | Main detection, tracking, lane, and orchestration pipeline |
| `visentra_live_accurate_low_latency.py` | Live-processing experiment |
| `visentra_live_optimized.py` | Live-processing optimization experiment |
| `visentra_virtual_corridor_events.py` | Corridor and event-processing experiments |
| `visentra_trajectory_predictor.py` | Kalman-based trajectory predictor |
| `evaluate_visentra_trajectories.py` | Trajectory evaluation utility, if present |
| `MonoPilot/` | Monocular depth and distance experiments |
| `Depth-Anything-V2/` | Depth Anything V2 source and experiments, if present |
| `CondLaneNet/` | Lane-model source and configuration, if present |
| `models/` | Local model checkpoints and exported models, if present |
| `test_images/` | Test images, if present |
| `test_videos/` | Recorded-video inputs, if present |
| `metric3d_test/` | Metric3D test outputs, if present |
| `research_models/` | Experimental model files, if present |
| `diagnose_*.py` | Diagnostic scripts |
| `benchmark_*.py` | Benchmarking experiments |

Large model weights, test media, generated results, backups, and virtual environments may be omitted from version control. A fresh clone may require separate model downloads, dependency installation, and path configuration.

## 15. Reproducing the Distance and Trajectory Experiments

The four reference images used in the documented distance test are:

```text
test_images/5m.png
test_images/10m.png
test_images/15m.png
test_images/20m.png
```

Use the current runner's help output and configuration to identify supported input and output options. Do not assume that every experimental script accepts the same arguments.

For reproducible distance tests, record:

- Reference camera-to-target distance.
- Selected vehicle detection and image region.
- Predicted depth or distance.
- Absolute error.
- Model and calibration configuration.
- Inference runtime and test conditions.

For trajectory experiments, record:

- Video identity and evaluated frame range.
- Track IDs and observation timestamps.
- Prediction horizon and model configuration.
- Predicted versus observed future positions.
- Number of valid predictions.
- Mean and median prediction errors.
- Missing observations and tracking interruptions.

Use the same evaluation clips and settings when comparing predictor changes.

## 16. Known Limitations

VISENTRA is a research prototype with the following limitations:

- **Distance accuracy:** The documented four-image test is too small to establish general accuracy.
- **CPU throughput:** Recorded CPU inference is substantially below the long-term real-time target.
- **Camera calibration:** Field of view, focal length, mounting height, and camera pitch affect geometric estimates.
- **Lane detection:** Markings may be faded, obscured, or confused with shadows and road edges.
- **Vehicle detection:** Occlusion, glare, low light, and motion blur may cause detection failures.
- **Tracking stability:** Missed detections, identity switches, and long gaps can disrupt track histories.
- **Trajectory prediction:** Forecasts depend on observation quality and may not capture sudden maneuvers.
- **Motion estimation:** Image displacement is not automatically equivalent to physical relative speed.
- **Generalization:** A few images or videos cannot demonstrate robustness across roads and weather.
- **Hardware variation:** Performance varies with the CPU, camera source, resolution, model, and software configuration.
- **Integration status:** Experimental scripts and separately installed models are not necessarily part of the active end-to-end pipeline.

## 17. Development Roadmap

### Phase 1 — Vehicle and Lane Perception
- Validate vehicle detections on repeatable road images and recorded videos.
- Evaluate lane-curve placement and visualization.
- Check the alignment of vehicle and lane outputs.
- Record failure cases and inference times.

### Phase 2 — Distance Estimation and Calibration
- Calibrate the actual camera.
- Expand known-distance tests.
- Evaluate depth and range errors separately.
- Improve target selection and geometric interpretation.
- Compare calibration changes using a fixed test set.

### Phase 3 — Tracking and Temporal Consistency
- Evaluate ByteTrack on recorded clips.
- Measure track fragmentation and identity switches.
- Examine track loss and recovery after occlusion.
- Check the behavior of the frame-order safeguard.
- Measure tracking latency and temporal gaps.

### Phase 4 — Trajectory Prediction
- Evaluate the Kalman-based predictor on recorded videos.
- Measure prediction error at several forecast horizons.
- Check sensitivity to missing observations and timestamp gaps.
- Compare HiVT only after establishing a reproducible evaluation setup.
- Keep experimental predictions separate from safety-critical decisions.

### Phase 5 — Live-Video Performance
- Stabilize camera input.
- Measure per-model and end-to-end latency.
- Investigate frame copies, queues, and unnecessary repeated processing.
- Compare live variants under identical conditions.
- Record dropped and skipped frames.

### Phase 6 — Virtual-Corridor and Event Evaluation
- Evaluate left, center, and right classification.
- Measure corridor-classification errors.
- Test event stability and false triggers.
- Evaluate multiple-vehicle scenarios.
- Keep experimental alerts separate from vehicle controls.

### Phase 7 — Broader Model and Road Validation
- Evaluate candidate models against the active pipeline.
- Test additional vehicles, roads, distances, and lighting conditions.
- Publish reproducible configurations and benchmark results.
- Document failures as well as successful examples.
- Reassess feasibility against the intended accuracy and latency requirements.

## 18. Safety Notice

VISENTRA is not a certified advanced driver-assistance system and must not be relied upon for safety-critical driving or reversing decisions.

Detection boxes, track identities, lane curves, distance estimates, and predicted trajectories may be incorrect. A visually plausible result does not guarantee accuracy.

Do not use VISENTRA as the sole basis for deciding whether a maneuver is safe, activating vehicle indicators, or controlling vehicle movement. Conduct experiments in controlled conditions without exposing people or property to risk.

## 19. Project Summary

VISENTRA explores camera-based vehicle detection, multi-frame tracking, lane perception, monocular depth estimation, trajectory prediction, and virtual-corridor event logic.

The principal components of the main pipeline are:

1. **RF-DETR** — vehicle detection.
2. **ByteTrack** — multi-frame vehicle tracking.
3. **CurveLaneS / CondLaneNet** — lane detection.
4. **Metric3D** — experimental monocular depth estimation.
5. **Kalman-based trajectory predictor** — experimental short-term motion forecasting.

The MonoPilot depth experiments and HiVT evaluation are separate workstreams and should be described according to their actual integration status.

The latest documented four-image distance experiment produced an MAE of approximately **2.38 m**. Recorded CPU tests have also shown substantial inference latency. These results provide a baseline for continued experimentation; they do not establish general distance accuracy, reliable trajectory prediction, or real-time safety performance.

The next priorities are broader distance validation, camera calibration, tracking stability, trajectory-prediction evaluation, and systematic measurement of end-to-end latency.

