# Hydrotester video detection

The same model and leak tracker process a recorded video or a USB camera. Each
session reports up to three confirmed leak positions and records through the
end of a clip or the application's stop command. A separate check reports
spreading water even when there are no countable droplets.

## Fabric boundary and false-detection checks

Recorded and live tests now locate the **inside edge of the white specimen rim**
automatically. The cyan outline in previews and saved images marks the allowed
area. Inference receives a crop with everything outside that oval masked; boxes
that cross the oval or are too large to be individual beads are rejected before
tracking. Calibration is checked on every analyzed frame. If the opening frame has no
reliable rim, the run reports an error and asks for a manual fabric area. If
calibration becomes unreliable later, detection pauses for those frames while
recording continues. There is no automatic fallback to the full image.

In **Detection settings**, choose automatic calibration or set a manual bounding
box around the fabric opening. The dashboard analyzes only the oval inside that
box. Existing API integrations can explicitly use `auto_fabric_area=False` for
other framing; manual `roi` coordinates take precedence over automatic detection.

YOLO runs also compare candidate patches with the opening frame to reject static
stitches, tape marks, and unchanged reflections, allowing small translations and
uniform brightness changes. Candidate boxes connected to a broad changed region
(over 20% of the analysis rectangle after smoothing) are also rejected to avoid
counting a spreading pool as an individual bead. This assumes the test starts with dry fabric. Turn
off **Ignore marks already present in the opening frame** if the video already
contains droplets; the CLI equivalent is `--include-opening-droplets`. Significant
movement rejects candidates, and the result includes a movement warning.
When a broad change masks a strong candidate, the detector can recheck it
against a recent frame without candidates. That check excludes positions the
model already proposed in the opening frame.

The dashboard uses one droplet model for recordings and the live camera. It uses
the latest successful dashboard training run when available, then the trained
visible-leak weights, then the general weights shipped with the repository. Both
YOLO and the command-line vision-language backend use the same area restriction.

Results JSON includes `fabric_roi` and `fabric_roi_shape` for checking calibration.

## Spreading water and pools

**Detect spreading water and pools** is enabled by default. This additional visual
check compares the fabric opening with its dry starting frame. It samples at
most four times per video second, requires substantial appearance change and a
new curved boundary in three consecutive, spatially consistent observations,
and compensates for small translations and uniform brightness changes. It masks
long opening-frame seams so pressure-induced seam bending is not counted as water.
Both the opening and current fabric boundaries restrict the accepted region.
Live capture performs the check on the inference worker. Vision-backend runs use
their configured, slower sampling interval.

This check works independently of YOLO proposals. It adds `surface_water` to
the result with first-observation and confirmation times, an observed boundary,
and `surface_water.jpg`. The amber boundary is a detected visual edge, not a
full water segmentation or the position where leakage began. It does not add
invented droplets to `leaks`. `leakage_detected` is true when either a droplet or
spreading-water event exists. Integrations should use that field for presence
and `leaks` for individual bead positions. The reference CSV comparison still
evaluates individual beads only.

The broad-pool rejection in bead counting remains enabled. Turn off the separate
surface-water checkbox, or pass `--no-surface-water` in either video/camera CLI,
when no dry opening is available. This is a conservative heuristic, not a newly
trained water segmentation model: diffuse films, water already present in the
opening, water without a clear curved edge, or large movement can be missed.
Reflections and deformation can still produce false warnings. Absence of a
warning does not prove the specimen is dry.

Local checks on 2026-11-21:

- Full dashboard pipeline on `100926-0505`: zero countable droplets, but a
  spreading-water warning at 108.37 s, confirmed at 108.88 s. All frames were
  recorded. Analysis took 107.29 s for the roughly 116-second recording.
- Final boundary-restricted surface checks on that clip and three other visible
  leak recordings (`100926-0923`, `101926-1207`, `101426-0932`) produced warnings;
  their snapshots were visually checked.
- Three complete recordings from the no-visible-leak folder (`102026-1115`,
  `101426-0519`, `101926-1411`) produced no surface-water warnings.
- A sparse development screen found no candidate boundaries in 640 prepared
  frames from 160 videos in that folder; 15 additional videos required manual
  rim calibration. That screen is not a full-video accuracy measurement.

Folder categories are weak labels and the visual reviews are provisional, not
expert ground truth. These checks do not establish general accuracy. Evidence
is under `data/results/surface_water_20261121/`; reproduce complete-recording
surface checks with `scripts/evaluate_surface_water.py VIDEO ... --output-dir NEW_DIR`.

## Faster local video analysis

The detector automatically uses a verified float32 ONNX export when one is
available for the exact selected weights. Verified Windows GPU execution uses
DirectML, with the verified CPU runtime available if that adapter cannot load.
This runs the same 640-pixel YOLO
network with the original confidence, suppression and coordinate rules, without
the slower general-purpose prediction wrapper. Temporal confirmation, the fabric
boundary, and opening-frame/pool checks remain enabled. The dashboard normally
examines up to six frames per video second. Enable **Analyze every frame** under
**Detection settings** when short-lived events are important. The Python and
command-line interfaces analyze every frame by default for recorded video.

Exports are stored in `data/results/inference_cache/`, indexed by the source
weights' SHA-256. The original `.pt` files remain available for training. A
missing, unverified, modified or incompatible export falls back to the original
runtime. Retrained weights cannot accidentally reuse an old export.

After training other weights, prepare their runtime using real frame examples:

```powershell
.\.venv\Scripts\python.exe scripts/optimize_inference.py models/droplet-seg.pt --validation-manifest data/raw/reviewed_candidate_20261121/test_selection.json
```

The manifest is a JSON array with an `image` path for each example. The command
checks both full images and masked specimen crops, requiring matching detection
counts, coordinates within 0.1 pixels and confidence within 0.0001 before
enabling an export. This is numerical parity checking, not a new accuracy claim.
Install `.[ml]` for the exporter and optional ONNX Runtime dependencies. Windows
x64 installs the DirectML build; other platforms use the CPU build. To verify
and enable a particular Windows graphics adapter, add `--gpu-device ID` to the
optimization command. This computer's faster adapter was verified as ID 1.
GPU predictions undergo the same numerical parity checks before being enabled.

On this computer, a 78-frame, 4.97-second segment took 51.35 seconds with the
original engine and 5.86 seconds with the verified GPU runtime (8.76x faster).
Both runs saved all 78 frames, evaluated the same 77 reliable-area frames, and
paused on the same one uncertain-area frame. The leak position and source-video
timestamps matched. This is a measured example, not a guarantee for every video;
see `data/results/performance_20261121/video_benchmark.json` for the measurements.

## Use the dashboard

Run `python scripts/run_ui.py` from the project folder. In **Recorded video**,
an uploaded H.264 MP4 or WebM can be played and scrubbed as soon as the upload
finishes. **Analyze video** runs detection in the background with a separate
progress bar. Results include the detection map, event timeline, and position
plot. **Live camera** uses a USB camera attached to the computer hosting the app.
Press **Stop test and save results** to finalize the files.

### Try the local vision-language detector

Install [Ollama for Windows](https://docs.ollama.com/windows), start it, and
download the image-capable model once:

```powershell
ollama pull qwen3-vl:2b-instruct
```

For the command line, add `--detector vision --vision-interval 2` to either
video command. For a custom oval, add `--roi X Y WIDTH HEIGHT --roi-shape ellipse`.
Python callers can set `VideoSettings(detector_backend="vision", match_distance_px=30)`.
To analyze a single BGR image, call `VisionLanguageDetector()(image)`.
Start with a dry specimen so the vision backend can use the first frame as its
reference. To keep Ollama offline, start it with `OLLAMA_NO_CLOUD=1`.

### Label video frames in the dashboard

1. Open **Train the Model** and upload a recording. The frame editor opens
   as soon as the upload is ready.
2. Use **Jump to time (seconds)**, **Back 5 s**, or **Forward 5 s** to move
   through the recording. Use **Previous frame** and **Next frame** for exact
   labels. Drag one tight rectangle around each visible droplet and click **Save boxes**.
   For a frame with no visible water droplets, click **Save as dry**.
3. Save at least two different frames with marked droplets. Include a mix of
   early droplets, growth, and dry fabric. You can add recordings from different
   fabrics to the same set. In-app training crops the saved frames to the same
   oval used during replay and adjusts the boxes to match. The app holds back
   later labelled frames from a selected recording. When training the combined
   collection, it can hold out a separate recording.
4. Click **Train the model**. The app checks held-back boxes and, for a selected
   recording, replays the source to verify a marked leak position. A successful
   run updates the detector used by **Recorded video** and **Live camera**. A
   candidate that fails the checks leaves the existing detector active. CVAT
   YOLO Detection ZIPs use the image framing supplied in the ZIP. Use separate
   recordings and specimens for final accuracy checks.

Saved annotations are stored locally under `data/fabric_annotations/general/`.
Reupload the same recording to edit its saved boxes. The active dashboard
checkpoint is stored in `data/results/dashboard/active_droplet.pt`.

### Prepare videos sorted into leak/no-visible-leak folders

Folder names describe recordings, not droplet positions. Before retraining the
detector, review selected frames and mark every visible individual droplet with
a bounding box. Review frames from both folders; a negative result from sparse
sampling does not establish that the entire recording is droplet-free.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_video_training.py --source "D:\Side Quests\TransitionX\Data" --output data/raw/company_review_20261121
```

Choose a new output folder for each preparation run. This extracts four frames
per unique recording at 5%, 40%, 75%, and 98% of its duration, with source paths
and timestamps in `frames.json` and contact sheets in `review_sheets/`.
Recording dates and exact duplicate video contents remain in the same split.
Fabric identity is unknown, so verify independent specimens before evaluating.

Import `cvat_train_images.zip`, `cvat_val_images.zip`, and
`cvat_test_images.zip` as separate CVAT tasks. Add the `water_droplet` label,
review all frames, and export Ultralytics YOLO Detection annotations. Only
explicitly reviewed droplet-free frames should become negative examples. Keep
test annotations separate from training and threshold selection. Sparse
sampling may miss first visibility; add reviewed onset frames as needed.

The preparation command creates no detection labels and performs no training.
It preserves the source recordings and existing model weights.

### Train an isolated candidate from reviewed frames

`scripts/train_reviewed_candidate.py` converts explicitly reviewed preview boxes
back to full-image YOLO labels and fine-tunes a separate candidate. The annotation
manifest preserves the source video, timestamp, recording group, and review
status. AI-reviewed annotations remain provisional and need expert checking
before being used for acceptance claims. This path does not label the rest of
the prepared frames automatically or change the dashboard's default weights.

```powershell
.\.venv\Scripts\python.exe scripts/train_reviewed_candidate.py --annotations data/raw/reviewed_candidate_20261121/annotations.json --output data/results/reviewed_candidate_20261121 --epochs 30
```

Choose a new output directory for each run. Compare the resulting
`training/weights/best.pt` against the original model using separate reviewed
test frames with `scripts/evaluate_reviewed_candidate.py`. That comparison
checks frame-level boxes at confidence 0.25 and IoU 0.5; it does not establish
full-video leak timing or count accuracy. Test videos and recording dates must
be separate from both training and validation.

For a candidate trained on specimen crops, add `--specimen-crop` when training
and evaluating. The prepared recordings use the normalized crop
`left=.267, top=.087, right=.753, bottom=.961`. For a 1920-by-1080 recording,
the equivalent existing video option is `--roi 513 94 933 944`. Check the
specimen position before applying that crop to another camera setup.

The first cropped experiment on 21 November 2026 used a generic pretrained
`yolo11n.pt` starting point, 25 training frames and seven validation frames:

```powershell
.\.venv\Scripts\python.exe scripts/train_reviewed_candidate.py --annotations data/raw/reviewed_candidate_20261121/annotations.json --output data/results/reviewed_crop_candidate_20261121 --base-model yolo11n.pt --specimen-crop --epochs 40 --patience 15
```

It stopped after 35 epochs, selecting epoch 20. At confidence 0.25 and IoU 0.5,
the candidate matched four of eight provisionally marked droplets in a selected
15-frame held-out sample. It missed four and returned no detections on seven
reviewed negative frames. The original detector, evaluated with the same crop,
matched none of the eight and produced 58 false boxes. These are small-sample
results against AI-reviewed labels, not a general accuracy claim. The candidate
was evaluated with the fabric boundary and opening-frame checks. It still missed
droplets. Its local weights and comparison are under
`data/results/reviewed_crop_candidate_20261121/`.

The results panel accepts an optional reference CSV with `x_px,y_px,time_s`
to show matched, false, and missed leaks and timing errors against the
operator's tolerances. Outputs are stored in `data/results/dashboard/`.

## Run a company recording

From the project folder, with the ML dependencies and trained weights installed:

```powershell
.\.venv\Scripts\python.exe scripts/run_video.py "data/company_video/test_01.mp4" --output-dir "data/results/test_01"
```

Use a new output folder for each run. If the machine started 2.5 seconds into
the recording, add `--test-start 2.5`. Otherwise times are relative to the
start of the clip. Add `--roi X Y WIDTH HEIGHT` to limit analysis to the
fabric. Coordinates in results remain relative to the full frame. Start with
the defaults and tune confidence, confirmation hits, and matching distance on
development recordings.

The output includes `results.json`, the recording, `final_marked.jpg`, and
one `leak_XX.jpg` for each confirmed leak. The dashboard reuses an uploaded
recording when the test starts at zero. Other runs save a new MP4. Markers
are saved on the images.

## Integrate with the hydrotester

Try the camera with `python scripts/run_camera.py --camera-id 0 --output-dir data/results/live_01`.
The sample uses Enter for Start and Stop. The machine application can invoke
the same API from its own test controls.

```python
from droplet_detector import FrameDetector, OutputPaths, VideoSettings, start_test

outputs = OutputPaths(
    video="run_42/recording.mp4",
    images="run_42/detection_images",
    results="run_42/results.json",
    final_image="run_42/detection_map.jpg",
)
settings = VideoSettings(
    model_path="models/droplet-seg.pt",
    resolution=(1920, 1080),
    frame_rate=30,
    confidence_threshold=0.25,
)

# The hydrotester loads the model before enabling Start.
detector = FrameDetector(settings.model_path, settings.confidence_threshold)
session = start_test(camera_id=0, outputs=outputs, settings=settings, prepared_detector=detector)

# The hydrotester invokes this on Stop, or if session.done reports early exit.
report = session.stop_test()
print(report.model_dump_json(indent=2))
```

For replay from Python, call `analyze_video(path, outputs=outputs,
settings=settings, test_start_s=0.0)`. A result contains status, leaks, file
paths, warnings, and errors. Detection time is the timestamp of the first
matched observation. Confirmation time is when the required number of frame
observations has been reached. Reported positions and snapshots come from the
confirmation frame. Leak numbers follow confirmation order. When several
leaks confirm on the same frame, their first-seen times break the tie.

Live timestamps use monotonic time relative to Start. Inference can skip
frames if it cannot keep up, while the capture loop continues recording.
Check those warnings on the actual computer. Camera startup, resolution, and
recording performance need verification on the hydrotester.

## Acceptance checks for the company

Annotate the first visible time and position of each of the first three
distinct leaks in independent runs from at least three representative
fabrics. Compare matched events against the agreed timing tolerance and
record position error, false positives, duplicate counts, and misses.
Include dry runs, reflections, lighting changes, fabric movement and a
growing leak. The model currently relies on learned appearance and temporal
persistence. Persistent reflections can still pass and visible droplets can
still be missed. Use representative reviewed examples for retraining if
these cases fail. Keep final evaluation recordings separate from training
and threshold tuning.
