# VISENTRA
## Intelligent Vehicle Proximity Warning System
## Vision-Based Vehicle, Lane, Distance, and Trajectory-Awareness Research Pipeline

VISENTRA is a computer vision research project exploring how a road-facing camera can be used to detect vehicles, track them across frames, estimate lane boundaries, investigate approximate vehicle distances, and predict short-term vehicle motion.

The project combines **RF-DETR** for vehicle detection, **ByteTrack** for multi-frame tracking, **CurveLaneS / CondLaneNet** for lane detection, **Metric3D** for experimental monocular depth estimation, and a **Kalman-filter-based trajectory predictor** for short-term motion forecasting.

The project also investigates **HiVT (Hierarchical Vector Transformer)** as an advanced trajectory-prediction approach. HiVT experiments use pretrained HiVT-64 and HiVT-128 configurations and investigate converting camera-based vehicle tracking observations into trajectory-model inputs.

VISENTRA additionally includes a structured **86 Settings Configuration**, comprising 9 ESP32 GPIO assignments and 77 software configuration settings. These settings organize hardware connections, operating modes, sensor parameters, LED and buzzer behavior, timing, and advanced system controls.

Additional depth-estimation and trajectory-prediction approaches are investigated separately. These include the MonoPilot depth-processing workstream and HiVT experiments, whose integration and validation status must be distinguished from the confirmed operation of the main perception components.

VISENTRA investigates camera-only road awareness using a phone camera or compatible webcam, without requiring radar or lidar as core inputs.

**Project status:** Research and prototyping. Detection, tracking, lane estimation, depth estimation, trajectory prediction, virtual-corridor logic, and hardware configuration have different levels of validation. The complete system is not safety-certified.

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
- **Trajectory prediction:** Experiment with forecasting short-term vehicle motion using Kalman-based prediction and HiVT.
- **Virtual corridor:** Investigate vehicle positions relative to a configurable image-space region.
- **Hardware configuration:** Organize ESP32 GPIO assignments, sensors, indicators, operating modes, and system behavior through the 86 Settings Configuration.
- **Event logic:** Explore timed responses to persistent vehicle observations and multiple-object scenarios.
- **Performance evaluation:** Measure inference speed, processing throughput, skipped frames, and latency.

The long-term objective is to develop a low-cost, camera-based road-awareness prototype while measuring its limitations under repeatable test conditions.

## 2. System Architecture

VISENTRA contains related but distinct perception, prediction, hardware-configuration, and experimentation workstreams.

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
        (Kalman / HiVT)       |            |
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
                 Camera-Track Adapter
                         |
              +----------+----------+
              |                     |
              v                     v
        Kalman Predictor       HiVT Adapter
              |                     |
              v                     v
       Short-Term Forecast    HiVT Worker
                                    |
                                    v
                             HiVT-64 / HiVT-128
                                    |
                                    v
                             Six Prediction Modes
                                    |
                                    v
                              Future Trajectories
                                    |
                                    v
                           VISENTRA Visualization
```

The Kalman-based predictor and HiVT have different architectures and input requirements.

HiVT requires trajectory and map-related tensor inputs. VISENTRA investigates converting camera-based tracked observations into a compatible representation.

The predictions are experimental and must not automatically be interpreted as physical vehicle motion, collision probability, or road-relative velocity.

### D. Hardware and configuration workstream

```text
                  VISENTRA Configuration
                            |
                            v
                    86 Settings Design
                            |
                 +----------+----------+
                 |                     |
                 v                     v
             9 GPIO Pins         77 Software Settings
                 |                     |
                 v                     v
          ESP32 Connections     Operating Parameters
                 |                     |
                 +----------+----------+
                            |
                            v
                  Configurable System
                      Behavior
                            |
                            v
                   Hardware Testing
```

The 86 Settings Configuration provides a structured way to organize hardware and software parameters. Individual settings must be verified against the implemented source code and hardware behavior.

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

VISENTRA includes an experimental trajectory predictor implemented in:

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

## 7. HiVT — Hierarchical Vector Transformer Trajectory Prediction

HiVT (Hierarchical Vector Transformer) is an advanced trajectory-prediction approach investigated for VISENTRA.

Unlike a basic motion extrapolation method, HiVT uses a learned trajectory-prediction architecture to model historical agent movement and estimate multiple possible future trajectories.

The purpose of introducing HiVT is to investigate whether a pretrained trajectory-prediction model can provide more expressive future-motion forecasts than a simple constant-acceleration predictor.

### 7.1 HiVT model configurations

VISENTRA's HiVT experiments investigate two pretrained model configurations:

- **HiVT-64**
- **HiVT-128**

The configurations are evaluated as alternative model sizes within the same trajectory-prediction architecture.

The experiments use pretrained checkpoints and associated configuration information, including `hparams.yaml` files.

The models are maintained separately from the RF-DETR, lane-detection, and Metric3D environments because HiVT has its own dependency and input requirements.

### 7.2 Proposed VISENTRA integration architecture

The camera-based trajectory-prediction workflow is:

```text
             Vivo V27 Rear Camera / Webcam
                         |
                         v
                       RF-DETR
                  Vehicle Detection
                         |
                         v
                      ByteTrack
                 Persistent Track IDs
                         |
                         v
               Timestamped Track History
                         |
                         v
                 VISENTRA HiVT Adapter
                         |
                         v
               Camera-to-Trajectory
                    Conversion
                         |
                         v
                   HiVT Worker
                         |
                +--------+--------+
                |                 |
                v                 v
             HiVT-64           HiVT-128
                |                 |
                +--------+--------+
                         |
                         v
                 Six Future Modes
                         |
                         v
                Predicted Trajectories
                         |
                         v
                VISENTRA Preview
```

RF-DETR and ByteTrack provide the vehicle detections and temporal track observations. The adapter converts the camera-based tracking information into the tensor representation required by HiVT.

The worker provides an isolated execution process for loading the pretrained model and processing prediction requests.

### 7.3 HiVT input preparation

HiVT expects structured trajectory and map-related inputs rather than raw vehicle bounding boxes.

The original model input structure includes:

- `x` — historical agent trajectory information.
- `positions` — agent positions.
- `edge_index` — relationships between agents.
- `padding_mask` — valid and padded observation information.
- `bos_mask` — beginning-of-sequence information.
- `rotate_angles` — trajectory orientation information.
- Lane-related tensors and indices.
- Lane vectors and associated geometric information.

VISENTRA's adapter investigates converting ByteTrack observations into these model inputs.

The camera-to-trajectory conversion includes:

1. Collecting observations associated with persistent vehicle track IDs.
2. Maintaining timestamped vehicle history.
3. Extracting image-space positions from tracked vehicle detections.
4. Preparing the required historical trajectory length.
5. Converting the observations into the model's expected tensor layout.
6. Preparing lane-related inputs required by the model.
7. Sending the prepared inputs to the HiVT worker.
8. Receiving future trajectory predictions for visualization.

The investigated configuration uses **20 historical steps and predicts 30 future steps**, with **six possible prediction modes**.

These values describe the experimental model configuration and prediction interface. They do not establish that every real-camera prediction is accurate.

### 7.4 HiVT worker architecture

The experimental worker architecture separates model execution from the main perception pipeline.

The intended worker responsibilities include:

- Loading the pretrained HiVT checkpoint.
- Initializing the model and its dependencies.
- Receiving prediction requests.
- Preparing and validating tensor inputs.
- Running model inference.
- Returning predicted trajectories.
- Reporting inference timing.
- Handling errors and invalid requests.
- Supporting clean worker shutdown.

The worker uses a persistent process and a newline-delimited JSON communication protocol in the experimental implementation.

This architecture is intended to prevent trajectory-prediction dependencies from unnecessarily interfering with the main detection and tracking environments.

### 7.5 HiVT project files

The experimental integration workstream includes:

| File or directory | Purpose |
|---|---|
| `Claude/visentra.py` | Experimental main application integration |
| `Claude/visentra_hivt_worker.py` | Isolated HiVT model execution |
| `Claude/visentra_hivt_adapter.py` | Camera-track conversion and model-input preparation |
| `research_models/HiVT/` | HiVT source and model-related resources |
| HiVT-64 checkpoint | Pretrained trajectory-prediction configuration |
| HiVT-128 checkpoint | Alternative pretrained trajectory-prediction configuration |
| `hparams.yaml` | Associated model configuration information |

The root and Claude implementations have been treated as separate integration candidates. Their worker and adapter behavior must be compared before obsolete prototypes are replaced or merged.

The exact paths may differ between local development checkouts.

### 7.6 HiVT prediction output

The experimental HiVT configuration produces six possible future trajectory modes.

These modes represent alternative predicted future paths rather than six guaranteed future movements.

The intended visualization includes:

- Historical vehicle trajectory.
- Current tracked vehicle position.
- Predicted future trajectory.
- Multiple prediction modes.
- Prediction horizon.
- Model inference information.

The objective is to display the predicted paths in the VISENTRA preview alongside the detected and tracked vehicles.

The model's prediction output should remain associated with the correct track ID and timestamp. Asynchronous results must be checked for stale or outdated predictions before visualization.

### 7.7 Experimental testing and validation status

Initial HiVT experiments have investigated:

- Loading the pretrained model.
- Checking checkpoint and configuration compatibility.
- Preparing synthetic trajectory histories.
- Validating tensor dimensions.
- Checking coordinate transformations.
- Testing the six-mode prediction output.
- Verifying that generated outputs are finite and executable.
- Investigating worker communication and process handling.

Synthetic tests provide evidence of structural execution, not real-road prediction accuracy.

The adapter's current camera-input approach has important limitations:

- Camera detections are image-space observations, not calibrated world coordinates.
- Image-space positions do not directly establish physical vehicle velocity.
- The experimental adapter has used structurally valid empty lane tensors, rather than a fully verified camera-derived lane-map representation.
- Required metadata and complete original model-input compatibility still require verification.
- The current coordinate representation does not establish a reliable mapping to road-relative physical distances.

Consequently, successful synthetic execution must not be presented as proof of accurate real-world trajectory forecasting.

### 7.8 HiVT and Kalman predictor relationship

The existing Kalman predictor remains a separate baseline.

HiVT is investigated as an additional learned prediction approach rather than being treated as an automatic replacement for the Kalman implementation.

The two approaches should be evaluated using identical recorded videos and comparable track histories wherever possible.

Comparison criteria include:

- Prediction accuracy.
- Mean and median displacement error.
- Performance at different prediction horizons.
- Prediction stability.
- Sensitivity to missing track observations.
- Inference latency.
- CPU resource usage.
- Prediction availability for individual tracked vehicles.

HiVT should only be considered a validated replacement or active production component after successful real-video testing and reproducible performance evaluation.

## 8. VISENTRA 86 Settings Configuration

VISENTRA includes a structured **86 Settings Configuration** designed to organize hardware connections and software behavior.

The configuration contains:

- **9 ESP32 GPIO pin assignments.**
- **77 software configuration settings.**

Its purpose is to provide a structured configuration system for sensor parameters, operating modes, indicators, timing, and advanced system behavior.

### 8.1 Configuration breakdown

| Configuration category | Number of settings |
|---|---:|
| ESP32 GPIO assignments | 9 |
| Feature enable/disable | 13 |
| Distance / sensor | 14 |
| Operating-mode controls | 3 |
| LED configuration | 11 |
| Buzzer configuration | 10 |
| LDR configuration | 5 |
| System timing | 9 |
| Advanced settings | 7 |
| Ultrasonic scheduling | 1 |
| Mode multipliers | 4 |
| **Total configuration entries** | **86** |

The 77 software settings are separate from the nine hardware GPIO assignments.

### 8.2 Hardware GPIO configuration

The GPIO configuration organizes the ESP32 hardware connections used in the prototype.

The current recorded prototype pin assignments include:

| Hardware component | ESP32 GPIO |
|---|---:|
| Left LED | 25 |
| Right LED | 26 |
| Switch | 14 |
| LDR | 15 |
| Left ultrasonic TRIG | 5 |
| Left ultrasonic ECHO | 18 |
| Right ultrasonic TRIG | 32 |
| Right ultrasonic ECHO | 33 |

The configuration design contains nine GPIO assignment entries. The recorded prototype connections above document the known pin mapping; the remaining assignment must be verified against the actual hardware configuration before documenting it as a specific connection.

### 8.3 Software configuration categories

The 77 software settings are organized into ten categories.

**Feature enable/disable — 13 settings**

Organizes feature-level controls to enable or disable supported functionality.

**Distance and sensor — 14 settings**

Organizes distance thresholds and sensor-related parameters.

**Operating-mode controls — 3 settings**

Provides controls for selecting and managing operating modes.

**LED configuration — 11 settings**

Organizes LED behavior, brightness, and indicator-related parameters.

**Buzzer configuration — 10 settings**

Organizes buzzer behavior and associated alert parameters.

**LDR configuration — 5 settings**

Organizes light-dependent resistor configuration and lighting-related behavior.

**System timing — 9 settings**

Organizes timing values used by system operations and event processing.

**Advanced settings — 7 settings**

Provides additional configurable parameters for advanced behavior.

**Ultrasonic scheduling — 1 setting**

Organizes the ultrasonic scheduling control.

**Mode multipliers — 4 settings**

Provides one multiplier for each operating mode.

### 8.4 Operating modes and multipliers

The configuration design includes four operating profiles.

| Operating mode | Multiplier |
|---|---:|
| Parking | 0.75× |
| Daily Road | 1.00× |
| Traffic | 0.80× |
| Long Drive | 1.20× |

These multipliers are configuration values for the respective operating modes. Their actual effect depends on the parameters to which they are applied in the implementation.

### 8.5 Configuration objectives

The 86 Settings Configuration is intended to:

- Organize hardware and software parameters centrally.
- Provide configurable sensor thresholds.
- Support multiple operating profiles.
- Control LED and buzzer behavior.
- Configure lighting-related behavior.
- Organize system timing and scheduling.
- Provide feature-level enable/disable controls.
- Make future feature expansion easier.
- Reduce unnecessary changes to core application logic when adjusting supported parameters.

### 8.6 Configuration validation

The 86-entry configuration represents the designed configuration structure.

Validation should confirm:

- All nine GPIO assignments match the hardware.
- All 77 software settings have defined meanings.
- Each setting has an appropriate default and valid range.
- Operating-mode multipliers affect the intended parameters.
- Changes are applied consistently.
- Hardware behavior matches the configured values.
- Unsupported or conflicting combinations are handled safely.

The existence of a configuration entry does not automatically establish that its associated functionality has been implemented or validated.

## 9. Separate Python Environments

The models have different dependency requirements, so VISENTRA uses separate Python environments rather than forcing all dependencies into one installation.

| Component | Environment or location | Purpose |
|---|---|---|
| RF-DETR | `environments/rfdetr_env/` | Vehicle detection |
| ByteTrack | Main pipeline environment | Vehicle tracking |
| CurveLaneS / CondLaneNet | `environments/condlanenet_env/` | Lane detection |
| Metric3D | `MonoPilot/.venv/` | Experimental depth estimation |
| Kalman predictor | Main pipeline code | Short-term motion forecasting |
| HiVT | `research_models/HiVT/` and associated experimental environment | Learned trajectory prediction |

The experimental HiVT worker uses an isolated execution environment to accommodate its model dependencies.

These paths describe the Windows development workspace. Other computers may require different paths, dependencies, or configuration.

## 10. MonoPilot — Monocular Depth and Distance Estimation

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

## 11. Known-Distance Validation

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

## 12. CPU Performance and Recorded-Video Testing

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
- Kalman prediction processing time.
- HiVT inference time and FPS.
- End-to-end throughput.
- Frame-to-display latency.
- Dropped, skipped, and out-of-order frames.

A window that displays frames successfully does not establish that all models are processing those frames at the same rate.

## 13. Live-Processing Variants

The repository contains experimental implementations for investigating different processing approaches.

| File | Purpose |
|---|---|
| `visentra.py` | Main application and model orchestration |
| `visentra_live_accurate_low_latency.py` | Experimental live-processing implementation |
| `visentra_live_optimized.py` | Experimental live-processing optimization |
| `visentra_virtual_corridor_events.py` | Virtual-corridor and event-processing experiments |
| `visentra_trajectory_predictor.py` | Kalman-based trajectory prediction |
| `Claude/visentra.py` | Experimental HiVT application integration |
| `Claude/visentra_hivt_worker.py` | Isolated HiVT execution |
| `Claude/visentra_hivt_adapter.py` | Camera-track to HiVT input conversion |
| `evaluate_visentra_trajectories.py` | Experimental trajectory-evaluation utility, if present in the checkout |

These variants should be tested independently. Their accuracy and performance must not be assumed equivalent.

The intended live workflow is to acquire camera frames, run detection, track vehicles, estimate lane curves, process available depth outputs, update trajectory forecasts, and display the results.

Camera connectivity and throughput depend on the device, resolution, and capture method.

## 14. Virtual Corridor and Event Logic

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

## 15. Other Candidate Models and Approaches

Additional models and approaches may be evaluated separately from the main pipeline.

### DeepSORT

DeepSORT is a candidate alternative to ByteTrack.

It can be investigated for multi-object tracking performance, identity consistency, and behavior under occlusion.

It should not be treated as an additional active tracker alongside ByteTrack unless a specific configuration explicitly runs both.

### MobileNetV3

MobileNetV3 is a candidate for secondary classification or feature-extraction experiments.

Its usefulness should be evaluated against the existing RF-DETR pipeline before adoption.

### Model evaluation requirements

Any candidate should be compared against the current implementation using the same or equivalent data, documented metrics, and reproducible configurations before it is adopted.

## 16. Repository Structure

The following paths describe principal files and folders in the development workspace. Not all local files are necessarily tracked by Git.

| Path | Purpose |
|---|---|
| `visentra.py` | Main detection, tracking, lane, and orchestration pipeline |
| `visentra_live_accurate_low_latency.py` | Live-processing experiment |
| `visentra_live_optimized.py` | Live-processing optimization experiment |
| `visentra_virtual_corridor_events.py` | Corridor and event-processing experiments |
| `visentra_trajectory_predictor.py` | Kalman-based trajectory predictor |
| `Claude/visentra.py` | Experimental HiVT integration application |
| `Claude/visentra_hivt_worker.py` | HiVT model worker |
| `Claude/visentra_hivt_adapter.py` | HiVT camera-track adapter |
| `research_models/HiVT/` | HiVT source and model resources |
| `MonoPilot/` | Monocular depth and distance experiments |
| `Depth-Anything-V2/` | Depth Anything V2 source and experiments, if present |
| `CondLaneNet/` | Lane-model source and configuration, if present |
| `models/` | Local model checkpoints and exported models, if present |
| `test_images/` | Test images, if present |
| `test_videos/` | Recorded-video inputs, if present |
| `metric3d_test/` | Metric3D test outputs, if present |
| `research_models/` | Experimental model files |
| `diagnose_*.py` | Diagnostic scripts |
| `benchmark_*.py` | Benchmarking experiments |

Large model weights, test media, generated results, backups, and virtual environments may be omitted from version control. A fresh clone may require separate model downloads, dependency installation, and path configuration.

## 17. Reproducing the Distance and Trajectory Experiments

The four reference images used in the documented distance test are:

```text
test_images/5m.png
test_images/10m.png
test_images/15m.png
test_images/20m.png
```

Use the current runner's help output and configuration to identify supported input and output options. Do not assume that every experimental script accepts the same arguments.

### Distance evaluation

For reproducible distance tests, record:

- Reference camera-to-target distance.
- Selected vehicle detection and image region.
- Predicted depth or distance.
- Absolute error.
- Model and calibration configuration.
- Inference runtime and test conditions.

### Kalman trajectory evaluation

Record:

- Video identity and evaluated frame range.
- Track IDs and observation timestamps.
- Prediction horizon and model configuration.
- Predicted versus observed future positions.
- Number of valid predictions.
- Mean and median prediction errors.
- Missing observations and tracking interruptions.

### HiVT evaluation

Record:

- HiVT configuration: HiVT-64 or HiVT-128.
- Checkpoint and associated configuration.
- Number of historical observations.
- Number of predicted future steps.
- Number of available prediction modes.
- Camera-track conversion settings.
- Input tensor dimensions and validation results.
- Lane-related input preparation.
- Prediction latency.
- Predicted versus observed future positions.
- Number of valid predictions.
- Mean and median displacement errors.
- Worker startup and execution behavior.

Use the same evaluation clips and settings when comparing predictor changes.

## 18. Known Limitations

VISENTRA is a research prototype with the following limitations:

- **Distance accuracy:** The documented four-image test is too small to establish general accuracy.
- **CPU throughput:** Recorded CPU inference is substantially below the long-term real-time target.
- **Camera calibration:** Field of view, focal length, mounting height, and camera pitch affect geometric estimates.
- **Lane detection:** Markings may be faded, obscured, or confused with shadows and road edges.
- **Vehicle detection:** Occlusion, glare, low light, and motion blur may cause detection failures.
- **Tracking stability:** Missed detections, identity switches, and long gaps can disrupt track histories.
- **Kalman prediction:** Forecasts depend on observation quality and may not capture sudden maneuvers.
- **HiVT input compatibility:** Camera-based image-space tracks require appropriate conversion into the model's expected trajectory representation.
- **HiVT map information:** Camera-derived lane and map inputs require further validation.
- **HiVT coordinate limitations:** Current experimental image-plane predictions are not verified physical or world-coordinate trajectories.
- **Trajectory accuracy:** Synthetic model execution does not establish real-road forecasting accuracy.
- **Motion estimation:** Image displacement is not automatically equivalent to physical relative speed.
- **Configuration validation:** The 86 Settings Configuration requires verification against actual hardware and software behavior.
- **Generalization:** A few images or videos cannot demonstrate robustness across roads and weather.
- **Hardware variation:** Performance varies with the CPU, camera source, resolution, model, and software configuration.
- **Integration status:** Experimental scripts and separately installed models are not necessarily part of the active end-to-end pipeline.

## 19. Development Roadmap

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

### Phase 4 — Kalman and HiVT Trajectory Prediction

- Evaluate the Kalman-based predictor on recorded videos.
- Validate HiVT-64 and HiVT-128 checkpoint loading.
- Verify camera-track conversion.
- Complete the model's required input and metadata preparation.
- Validate lane-related inputs.
- Evaluate the six prediction modes.
- Integrate prediction visualization into the main preview.
- Measure prediction error at several forecast horizons.
- Compare HiVT and Kalman predictions using identical clips.
- Measure CPU inference latency and resource usage.
- Keep experimental predictions separate from safety-critical decisions.

### Phase 5 — 86 Settings Configuration

- Verify all nine ESP32 GPIO assignments.
- Define and validate all 77 software settings.
- Check feature enable/disable behavior.
- Validate distance and sensor parameters.
- Test operating-mode selection.
- Validate LED and buzzer configuration.
- Test LDR-related settings.
- Check system timing and ultrasonic scheduling.
- Verify the four operating-mode multipliers.
- Test configuration changes under controlled conditions.

### Phase 6 — Live-Video Performance

- Stabilize camera input.
- Measure per-model and end-to-end latency.
- Investigate frame copies, queues, and unnecessary repeated processing.
- Compare live variants under identical conditions.
- Record dropped and skipped frames.

### Phase 7 — Virtual-Corridor and Event Evaluation

- Evaluate left, center, and right classification.
- Measure corridor-classification errors.
- Test event stability and false triggers.
- Evaluate multiple-vehicle scenarios.
- Keep experimental alerts separate from vehicle controls.

### Phase 8 — Broader Model and Road Validation

- Evaluate candidate models against the active pipeline.
- Test additional vehicles, roads, distances, and lighting conditions.
- Publish reproducible configurations and benchmark results.
- Document failures as well as successful examples.
- Reassess feasibility against the intended accuracy and latency requirements.

## 20. Safety Notice

VISENTRA is not a certified advanced driver-assistance system and must not be relied upon for safety-critical driving or reversing decisions.

Detection boxes, track identities, lane curves, distance estimates, and predicted trajectories may be incorrect. A visually plausible result does not guarantee accuracy.

Do not use VISENTRA as the sole basis for deciding whether a maneuver is safe, activating vehicle indicators, or controlling vehicle movement. Conduct experiments in controlled conditions without exposing people or property to risk.

## 21. Project Summary

VISENTRA explores camera-based vehicle detection, multi-frame tracking, lane perception, monocular depth estimation, trajectory prediction, virtual-corridor event logic, and configurable hardware behavior.

The principal perception and prediction components are:

1. **RF-DETR** — vehicle detection.
2. **ByteTrack** — multi-frame vehicle tracking.
3. **CurveLaneS / CondLaneNet** — lane detection.
4. **Metric3D** — experimental monocular depth estimation.
5. **Kalman-based trajectory predictor** — experimental short-term motion forecasting.
6. **HiVT-64 / HiVT-128** — advanced trajectory-prediction experiments using historical observations and multiple future prediction modes.

The HiVT workstream investigates converting camera-based ByteTrack histories into the structured inputs required by a pretrained trajectory-prediction model. Its synthetic execution and structural tests must be distinguished from verified real-road inference and prediction accuracy.

MonoPilot remains the separate monocular depth and distance-estimation workstream.

The hardware and software configuration design contains **86 total settings**, comprising 9 ESP32 GPIO assignments and 77 software settings. It organizes feature controls, sensor parameters, operating modes, LED and buzzer behavior, LDR settings, timing, advanced controls, ultrasonic scheduling, and mode multipliers.

The latest documented four-image distance experiment produced an MAE of approximately **2.38 m**. Recorded CPU tests have also shown substantial inference latency. These results provide a baseline for continued experimentation; they do not establish general distance accuracy, reliable trajectory prediction, or real-time safety performance.

The next priorities are broader distance validation, camera calibration, tracking stability, HiVT input compatibility, real-video trajectory-prediction evaluation, 86-setting hardware and software validation, and systematic measurement of end-to-end latency.

**VISENTRA's long-term objective is to investigate a configurable, camera-based vehicle-awareness platform that combines computer vision, temporal tracking, depth analysis, trajectory forecasting, and structured hardware control.**
