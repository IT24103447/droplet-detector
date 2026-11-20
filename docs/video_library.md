# Hydrotester video detection

The same model and leak tracker process a recorded video or a USB camera. Each
session reports up to three confirmed leak positions and records through the
end of a clip or the application's stop command.

## Use the dashboard

Run `python scripts/run_ui.py` from the project folder. **Recorded video**
accepts a local recording and displays the detection map, event timeline,
position plot and results. **Live camera** uses a USB camera attached to the
computer hosting the app. Press **Stop test and save results** to finalize
the files. **Fabric models** trains a separate selectable profile. Choose
**Label video frames** to work entirely in the dashboard, or import an
existing labelled CVAT YOLO Detection ZIP. The general model remains
available for detection.

### Label video frames in the dashboard

1. Enter a fabric profile name and choose **Training**. Upload a recording
   and click **Open in frame editor**.
2. Jump to a frame or use **Previous frame** and **Next frame**. Drag one
   tight rectangle around each visible droplet and click **Save boxes**.
   For a frame with no visible water droplets, click **Save as dry**.
3. Save selected frames with a mix of early droplets, growth, and dry
   fabric. Only saved frames enter the dataset. Choose **Validation** and
   open a different recording of the same fabric. Label representative
   frames there too. The same uploaded recording cannot be saved in both
   splits.
4. Click **Train fabric model**. When training finishes, the profile appears
   in the model picker for recorded and camera tests. Repeat with a new
   profile name for a different fabric. Hold back separate runs for final
   accuracy checks.

Saved annotations are stored locally under `data/fabric_annotations/`.
Reopen the same file under the same fabric name to edit its saved boxes.

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

The output includes `results.json`, `recording.mp4`, `final_marked.jpg`,
and one `leak_XX.jpg` for each confirmed leak. The recording is original
camera imagery re-encoded as MP4. Markers are saved on the images.

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
