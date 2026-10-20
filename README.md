

Readme · MD
VISENTRA
Vision-Based Vehicle, Lane, and Distance-Awareness Research Pipeline
VISENTRA is a computer vision project exploring how a road-facing camera can be used to detect vehicles, track them across frames, identify lane boundaries, estimate approximate vehicle distances, and investigate camera-based driving-assistance features.

The project combines RF-DETR for vehicle detection, ByteTrack for multi-frame vehicle tracking, CurveLaneS / CondLaneNet for lane detection, and a separate MonoPilot experimental pipeline for monocular depth estimation and distance analysis.

The project is designed to investigate camera-only road awareness — using a phone camera or a compatible webcam — without relying on additional radar or lidar sensors.

Project status: Research and prototyping. Vehicle detection, tracking, lane detection, distance estimation, live-video processing, and virtual-corridor event logic have different levels of validation. The complete system is not safety-certified.

1. Why VISENTRA?
Vehicles often need awareness of nearby traffic, road boundaries, and the relative positions of other vehicles. Dedicated sensors can provide distance measurements, but they can also increase hardware cost and integration complexity.

VISENTRA investigates how much useful information can be obtained from a camera alone.

The main objectives are:

Vehicle detection: Identify vehicles visible in the camera frame.
Vehicle tracking: Associate vehicle detections across frames to maintain consistent identities over time.
Lane detection: Estimate road-lane curves and visualize the apparent driving corridor.
Distance estimation: Investigate approximate vehicle range using monocular depth estimation.
Relative-position analysis: Explore whether detected vehicles are positioned to the left, right, or center of the camera view.
Virtual corridor: Experiment with defining a road region and detecting objects within or around it.
Event logic: Investigate timed responses to vehicle positions and multiple-object scenarios.
Low-cost implementation: Test the models using a Windows laptop and a phone camera or compatible webcam.
The long-term objective is to develop a camera-based road-awareness prototype that can provide useful visual information while keeping hardware requirements low.

2. System Architecture
VISENTRA contains two related but separate computer vision workstreams.

A. Vehicle detection, tracking, and lane detection
text
                       VISENTRA
                          |
                          v
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
             Multi-Frame Vehicle Tracking
                          |
                          v
                 Tracked Vehicle IDs
                          |
             +------------+------------+
             |                         |
             v                         v
      Tracked Vehicles          CurveLaneS /
                                  CondLaneNet
                                 Lane Detection
             |                         |
             v                         v
       Vehicle Positions          Lane Curves
             |                         |
             +------------+------------+
                          |
                          v
                   Result Fusion /
                 Visualization of
                Available Outputs
B. Monocular distance estimation
text
                Camera Image
                     |
                     v
             Vehicle Detection
                     |
                     v
              Vehicle Bounding Box
                     |
                     v
           Metric Depth Estimation
                     |
                     v
          Depth and Geometry Analysis
                     |
                     v
           Approximate Range Estimate
                     |
                     v
          Ground-Truth Error Evaluation
The two workstreams address different problems. Detection, tracking, and lane models provide object, identity, and road information, while MonoPilot investigates whether a single camera can provide useful metric-distance estimates — as a separate, experimental depth-processing workstream. Not every component runs simultaneously, and not every output listed above has the same level of validation; see the sections below for what has been tested.

3. Vehicle Detection — RF-DETR
RF-DETR is used to detect vehicles in road images and video frames.

Its output includes detected objects, bounding boxes, and the class and confidence information supported by the model.

Why RF-DETR?
Vehicle detection provides the object locations needed for subsequent analysis. For example, a detected vehicle's image position can be used to investigate whether it appears to the left, right, or center of the camera view, and its bounding box is the input that ByteTrack associates across frames.

Current implementation
RF-DETR model integration.
Separate Python environment for dependency compatibility.
Worker-based execution in the main pipeline.
Experimental live-processing variants.
Vehicle bounding-box visualization.
Diagnostic scripts for checking detection behavior.
Detection accuracy and inference speed must be measured under the intended camera conditions.

4. Vehicle Tracking — ByteTrack
ByteTrack is used in the main pipeline to associate RF-DETR vehicle detections across successive frames.

Why tracking?
A detector alone reports vehicles independently in every frame, with no notion of which detection in frame N corresponds to which detection in frame N+1. Tracking addresses this by:

Maintaining vehicle identities over time — assigning a consistent ID to the same physical vehicle as it moves across frames, instead of treating it as a new object each time.
Supporting temporal analysis — downstream experiments that need to reason about a vehicle's behavior over several frames (rather than a single snapshot) depend on having a stable identity to follow.
Feeding the virtual-corridor and event-processing experiments — visentra_virtual_corridor_events.py uses tracked vehicle identities, instead of raw per-frame detections, to investigate whether a given vehicle persists within or near the defined corridor over time and to drive event timing.
Limitations
Missed detections: If RF-DETR fails to detect a vehicle in a given frame, ByteTrack has no detection to associate, which can break or shorten a track.
ID switches: Two vehicles passing close together, or a vehicle briefly leaving and re-entering the frame, can cause the tracker to swap or reassign identities.
Occlusion: A vehicle temporarily hidden behind another object can lose its track or be reassigned a new ID when it reappears.
Camera movement: Camera motion (e.g., a handheld phone or a vehicle-mounted setup) changes the apparent motion of all objects in the frame, which can affect association quality.
Tracking instability: Track IDs, bounding boxes, and track lifetimes can fluctuate between frames, particularly at lower detection confidence or frame rates.
Tracking alone does not measure accurate distance, relative speed, or collision risk. It provides consistent identities and image-space positions over time; any distance, speed, or risk claim requires the separate depth-estimation and calibration work described below, combined with careful temporal analysis — none of which is implied by tracking by itself.

5. Lane Detection — CurveLaneS / CondLaneNet
CurveLaneS / CondLaneNet is used to estimate lane curves from road images.

The estimated curves can be overlaid on the camera frame to visualize the apparent road corridor.

Why lane detection?
Lane information provides road context that vehicle detection and tracking alone cannot supply. It can help investigate whether a tracked vehicle appears inside, outside, or near a defined corridor.

Current implementation
CondLaneNet model integration.
CurveLaneS configuration and model loading.
Separate Python environment for compatibility with the lane model's dependencies.
Lane inference and curve visualization.
Experimental integration with vehicle detections.
Lane detection is affected by lighting, road markings, shadows, occlusion, camera angle, and road geometry. Estimated lane curves should not automatically be treated as exact road boundaries.

6. Separate Python Environments
The perception models have different dependency requirements. VISENTRA therefore uses separate environments rather than attempting to install every model into one Python environment.

Component	Environment or location	Purpose
RF-DETR	environments/rfdetr_env/	Vehicle detection
ByteTrack	Main pipeline environment	Multi-frame vehicle tracking
CondLaneNet	environments/condlanenet_env/	Lane detection
MonoPilot	MonoPilot/.venv/, if configured	Depth and distance experiments
The main application coordinates the model workers and combines their outputs.

This separation helps prevent dependency conflicts and allows individual models to be tested independently.

The paths shown are from the current Windows development setup. They may need adjustment on another computer.

7. MonoPilot — Monocular Depth and Distance Estimation
MonoPilot is the experimental distance-estimation workstream under MonoPilot/.

It investigates whether a single camera can estimate the approximate distance to a detected vehicle without a dedicated distance sensor. This workstream is separate from the main RF-DETR / ByteTrack / lane-detection pipeline described above.

Main components
YOLO11n ONNX: Vehicle detection.
Metric Depth Anything V2 ONNX: Monocular metric-depth prediction.
Geometric processing: Analysis of depth and image-coordinate relationships.
Calibration options: Configurable parameters used to investigate distance errors.
Diagnostic tools: Scripts for examining model outputs, vehicle positions, depth values, and estimated ranges.
Benchmark tools: Experiments for evaluating inference speed and processing behavior.
Why combine detection and depth estimation?
A vehicle detector identifies an object and its approximate image location. A depth model provides depth information for the image.

Combining these outputs allows the project to investigate the distance to a detected vehicle rather than simply identifying its presence.

However, a model's depth output must be interpreted correctly. Camera-axis depth, commonly represented as Z, is not necessarily the same as the full straight-line distance between the camera and the target.

The estimated range may also be affected by the selected vehicle pixels, bounding-box placement, camera calibration, target occlusion, and depth-model errors.

8. Latest Known-Distance Validation
The latest recorded MonoPilot validation used four still images with known approximate vehicle distances.

Test image	Reference distance	Estimated distance	Absolute error
5m.png	5 m	7.972 m	2.972 m
10m.png	10 m	13.088 m	3.088 m
15m.png	15 m	16.356 m	1.356 m
20m.png	20 m	22.090 m	2.090 m
Mean absolute error	—	—	2.377 m
The mean absolute error is calculated as:

M
A
E
=
1
N
∑
i
=
1
N
∣
d
i
p
r
e
d
i
c
t
e
d
−
d
i
r
e
f
e
r
e
n
c
e
∣
MAE= 
N
1
​
  
i=1
∑
N
​
  
​
 d 
i
predicted
​
 −d 
i
reference
​
  
​
 
For the four recorded observations:

M
A
E
=
2.972
+
3.088
+
1.356
+
2.090
4
≈
2.377
 m
MAE= 
4
2.972+3.088+1.356+2.090
​
 ≈2.377 m
What these results mean
The estimated distances increased with the reference distances across the four tested images.
All four estimates were greater than their reference distances.
The mean absolute error was approximately 2.38 metres.
The 15-metre reference image had the lowest absolute error in this set.
The results are useful for debugging but do not establish reliable distance measurement across general road scenes.
This is a small validation set. It is not a statistically representative benchmark, and it does not establish safety for vehicle operation.

What still needs improvement?
Camera-specific calibration.
Consistent vehicle detection and target selection.
Reliable conversion from model depth to the required distance measurement.
Testing across more reference distances and vehicle types.
Evaluation under different camera positions and road conditions.
Repeated measurements and statistical error analysis.
The same calibration settings should be used throughout a test series unless a change is explicitly documented.

9. CPU Performance and Live-Video Testing
The current development setup uses a Windows 11 laptop with an Intel Core i5-1335U processor and no dedicated GPU.

The recorded MonoPilot outdoor CPU tests ran at approximately 1.3–1.5 FPS.

This is an offline experimentation result, not a confirmed live-camera frame rate, and it does not include RF-DETR detection or ByteTrack tracking overhead from the main pipeline.

The project has also encountered camera-capture performance limitations during earlier live tests. Camera acquisition, image conversion, model inference, tracking association, result fusion, and display can each contribute to latency.

Performance goals
The longer-term target is to investigate 30–60 FPS processing, but this has not been demonstrated by the current recorded CPU results.

Future performance reports should distinguish:

Camera capture FPS.
Vehicle-detection inference time.
Tracking association time.
Lane-detection inference time.
Depth-estimation inference time.
End-to-end processing FPS.
Frame-to-display latency.
Dropped or skipped frames.
A live window that displays frames successfully does not, by itself, establish that the full pipeline meets real-time requirements.

10. Live Processing Variants
The repository contains experimental implementations intended to investigate different processing approaches.

File	Purpose
visentra.py	Main VISENTRA application and orchestration (RF-DETR, ByteTrack, lane detection).
visentra_live_accurate_low_latency.py	Experimental live-processing implementation focused on accuracy and latency.
visentra_live_optimized.py	Experimental live-processing optimization.
visentra_virtual_corridor_events.py	Experimental virtual-corridor and event-processing logic, built on tracked vehicle identities.
These scripts represent development variants. They should be tested independently before their outputs or performance are considered equivalent.

Live-camera objective
The intended workflow is:

Obtain frames from a compatible camera.
Run vehicle detection.
Track vehicles across frames.
Estimate lane curves.
Display the detections, tracks, and lane overlays.
Investigate vehicle position and distance over successive frames.
Measure latency and check for unstable detections or tracks.
The source camera may be a phone's rear-facing camera or another supported webcam. Camera connectivity and frame rate depend on the capture method and device configuration.

11. Virtual Corridor and Event Logic
visentra_virtual_corridor_events.py explores the concept of a virtual corridor around the visible road region.

The idea is to use image-space vehicle positions, tracked identities, and estimated road context to investigate possible responses to nearby vehicles.

Intended experiments
Determine whether a tracked vehicle appears on the left, right, or center of the camera view.
Investigate whether the vehicle is within a configured virtual corridor.
Track observations over time (via ByteTrack identities) instead of making decisions from one frame.
Experiment with event timing for persistent or multiple vehicle detections.
Investigate how distance thresholds affect event decisions.
Why use a virtual corridor?
A corridor provides a structured region for interpreting tracked vehicle positions relative to the apparent path of the host vehicle. It can help distinguish an object near the visible path from one farther to the side.

However, image position is not equivalent to a vehicle's actual road position, and the apparent corridor is not a calibrated three-dimensional safety zone.

Important limitations
A single camera does not directly measure the motion or trajectory of another vehicle.
Relative speed cannot be reliably inferred from one image, and is not reliably established by tracking alone.
Monocular motion estimation requires temporal information, calibration, and careful treatment of camera movement.
Detection jitter, ID switches, occlusion, road curvature, and camera mounting can change the apparent vehicle position.
Event logic must account for uncertainty and false detections.
Any indicator or warning behavior remains experimental. It must not control actual vehicle movement or replace the driver's direct observation.

12. Proposed Models and Future Evaluation
MobileNetV3 and DeepSORT are candidates for future evaluation, not confirmed integrated components of the current pipeline.

MobileNetV3
Proposed potential role: lightweight secondary vehicle classification or feature extraction on top of the existing RF-DETR detections (for example, refining a vehicle category or extracting features for downstream use), to be adopted only if it is shown to add measurable value beyond what the existing detector already provides.

DeepSORT
Proposed as a possible alternative to ByteTrack for multi-frame vehicle tracking. DeepSORT and ByteTrack are not intended to run simultaneously in the main pipeline; any adoption of DeepSORT would be evaluated as a replacement candidate, not an addition.

Evaluation requirements
Any proposed model (MobileNetV3, DeepSORT, or others considered later) must be evaluated against the existing pipeline using comparable test data and metrics before being adopted. This includes:

Using the same or equivalent test images/video as the current RF-DETR and ByteTrack components.
Reporting accuracy, latency, and resource-usage differences relative to the current implementation.
Confirming the candidate model's dependencies integrate cleanly with the project's existing Python environments.
No dependencies have been installed, and no application code has been modified, in order to produce this documentation — these models are documented as proposals only.

13. Repository Structure
The following describes the principal files and directories used in the current development workspace. Some local files may not be included in the GitHub repository.

Path	Purpose
visentra.py	Main vehicle detection, tracking, and lane perception pipeline.
visentra_live_accurate_low_latency.py	Live-processing experiment.
visentra_live_optimized.py	Optimized live-processing experiment.
visentra_virtual_corridor_events.py	Virtual-corridor and event experiments.
MonoPilot/	Monocular depth and distance-estimation workstream.
Depth-Anything-V2/	Depth Anything V2 source and development files, if present.
CondLaneNet/	CondLaneNet source and configuration, if present.
models/	Local model checkpoints and exported models, if present.
test_images/	Test images and videos, if present.
inference_results/	Generated inference results, if present.
outputs/	Experimental output files, if present.
diagnose_*.py	Diagnostic scripts.
inspect_*.py	Model and output inspection scripts.
benchmark_*.py	Performance benchmarking experiments.
depth_calibration_config.json	Local distance-calibration configuration, if present.
distance_calibration_rfdetr.json	Experimental calibration configuration, if present.
A file being present locally does not necessarily mean it is tracked by Git.

Large model weights, test media, generated output directories, and virtual environments may be omitted from version control. A fresh clone may require additional model files, dependency installation, and path configuration.

14. Reproducing the Distance Tests
The four reference images used in the latest recorded test are:

test_images/5m.png
test_images/10m.png
test_images/15m.png
test_images/20m.png
The exact test command depends on the current MonoPilot configuration and runner options. Use the current MonoPilot runner's help output and configuration to select the image directory, outdoor settings, and output directory.

A reproducible test should record:

The measured camera-to-target reference distance.
The selected vehicle detection.
The predicted depth or distance.
The absolute error.
The model and calibration configuration.
The inference runtime.
The output image and diagnostic log.
Repeat the tests across more distances, vehicles, lighting conditions, and camera angles. Preserve the outputs so that changes to calibration or inference code can be compared fairly.

15. Known Limitations
VISENTRA is a research prototype. The main limitations include:

Distance accuracy: The four-image test set is too small to establish general accuracy.
CPU throughput: Recorded depth-testing performance is substantially below the desired real-time target.
Camera calibration: Field of view, focal length, mounting height, and pitch affect geometric estimates.
Lane detection: Lane markings can be faded, obscured, missing, or confused with road edges and shadows.
Vehicle detection: Detection may fail under occlusion, low light, glare, unusual viewpoints, or motion blur.
Tracking stability: ByteTrack identities can be affected by missed detections, ID switches, occlusion, and camera movement; temporal consistency requires further evaluation.
Motion estimation: Image displacement alone — whether from raw detections or tracked identities — does not establish reliable relative speed or collision risk.
Generalization: Results on a few test images do not demonstrate robustness across real roads and weather.
Hardware variation: Performance depends on the CPU, camera source, resolution, model, and software configuration.
16. Development Roadmap
Phase 1 — Vehicle and Lane Perception
Verify vehicle detections on repeatable road images and recorded videos.
Validate lane curves and their visualization.
Check how both outputs align in the same frame.
Document model failures and inference time.
Phase 2 — Distance Estimation and Calibration
Calibrate the actual camera.
Expand known-distance tests.
Evaluate depth and range errors separately.
Improve target selection and geometric interpretation.
Compare calibration changes on the same fixed test set.
Testing goals: distance error (mean/median absolute error), error by reference distance, calibration repeatability.
Phase 3 — Tracking and Temporal Consistency
Evaluate ByteTrack performance on recorded clips with known vehicle counts.
Measure ID-switch frequency and track-loss/recovery behavior under occlusion and camera movement.
Compare tracking behavior across different detection confidence thresholds.
Testing goals: tracking stability (ID-switch rate, track fragmentation), false detections feeding into tracks, end-to-end latency added by tracking.
Phase 4 — Live-Video Performance
Stabilize the camera input.
Measure capture FPS and end-to-end latency, broken down by pipeline stage.
Reduce unnecessary frame copies and repeated processing where appropriate.
Compare the live-processing variants under identical conditions.
Testing goals: inference FPS per model, end-to-end FPS, frame-to-display latency, dropped/skipped frames, CPU utilization.
Phase 5 — Virtual-Corridor and Event Evaluation
Evaluate tracked-vehicle corridor classification across recorded clips.
Test left, center, and right classification accuracy.
Measure event stability and false-trigger frequency.
Test multiple-vehicle cases and time-based event logic.
Keep event decisions separate from any safety-critical vehicle controls.
Testing goals: false-trigger rate, event timing accuracy, stability under multiple simultaneous vehicles.
Phase 6 — Proposed Model Evaluation and Broader Validation
Evaluate MobileNetV3 as a candidate for secondary classification/feature extraction, against the existing detector on the same test data.
Evaluate DeepSORT as a candidate alternative to ByteTrack, against the same tracking test set and metrics.
Test more road types, vehicles, distances, and lighting conditions.
Document failures as well as successful examples.
Publish reproducible configurations and benchmark results.
Reassess the project's feasibility against the intended latency and accuracy requirements.
Testing goals: head-to-head accuracy/latency comparisons for each proposed model, false detections, distance error, inference FPS, end-to-end latency, CPU utilization.
17. Safety Notice
VISENTRA is not a certified advanced driver-assistance system and must not be relied upon for safety-critical driving or reversing decisions.

The current distance estimates, tracking behavior, and virtual-corridor logic are experimental. A visually plausible bounding box, track, lane curve, or distance value does not guarantee that it is correct.

Do not use the system as the sole basis for activating vehicle indicators, deciding whether a maneuver is safe, or controlling vehicle movement. All experiments involving a vehicle should be performed in controlled conditions without placing people or property at risk.

18. Project Summary
VISENTRA explores camera-based vehicle detection, multi-frame vehicle tracking, lane perception, monocular depth estimation, and virtual-corridor event logic.

The project combines RF-DETR, ByteTrack, and CurveLaneS / CondLaneNet for vehicle, tracking, and lane perception, while MonoPilot investigates approximate distance estimation using vehicle detection, metric depth prediction, and geometric analysis as a separate experimental workstream. MobileNetV3 and DeepSORT are proposed candidates for future evaluation, not confirmed integrated components.

The latest recorded four-image distance experiment achieved a mean absolute error of approximately 2.38 m, while the recorded outdoor CPU depth tests ran at approximately 1.3–1.5 FPS. These measurements establish a starting point for further development, not a guarantee of general accuracy or real-time performance.

The next priorities are broader distance validation, camera calibration, stable live-video processing, tracking stability evaluation, and systematic evaluation of temporal, corridor, and proposed-model behavior.


Claude finished the response
