"""Lightweight operator dashboard for the hydrotester library."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import itertools
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

import altair as alt
import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from droplet_detector import OutputPaths, VideoSettings, analyze_video, start_test
from droplet_detector.fabric_training import train_annotated_model, train_profile
from droplet_detector.video import VideoPreview
from droplet_detector.video_annotation import label_path, load_boxes, read_frame, save_frame, video_info

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "droplet-seg.pt"
RUNS = ROOT / "data" / "results" / "dashboard"
TRAINED_MODEL = ROOT / "models" / "fabrics" / "Visible_Leaks_Trained.pt"
ACTIVE_MODEL = RUNS / "active_droplet.pt"
ANNOTATIONS = ROOT / "data" / "fabric_annotations" / "general"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]
PREVIEW_FORMATS = {"mp4": "video/mp4", "mov": "video/mp4", "webm": "video/webm"}
FAST_ANALYSIS_FPS = 6.0
TEAL = "#087F79"
box_editor = components.declare_component("droplet_boxes", path=str(ROOT / "ui" / "box_editor"))


class Preview:
    """A worker replaces the latest preview; only Streamlit renders it."""

    latest: VideoPreview | None = None

    def publish(self, frame: VideoPreview) -> None:
        self.latest = frame


class ReplayProgress:
    seconds: float | None = None

    def publish(self, elapsed: float) -> None:
        self.seconds = elapsed


@st.cache_resource
def workers():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="hydrotester")


@st.cache_resource
def training_lock():
    return Lock()


def active_model() -> Path:
    return next((path for path in (ACTIVE_MODEL, TRAINED_MODEL) if path.is_file()), DEFAULT_MODEL)


def save_upload(upload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    upload.seek(0)
    try:
        with pending.open("xb") as target:
            shutil.copyfileobj(upload, target)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def busy() -> bool:
    return (training_lock().locked()
            or any(job is not None and not job.done()
                for job in (st.session_state.get("replay"), st.session_state.get("training")))
            or (st.session_state.get("live") is not None and not st.session_state.live.done))


def fabric_area_setting(key: str):
    manual = st.radio("Fabric area", ("Find inside the white rim automatically", "Set the fabric area manually"),
                      key=f"{key}_fabric_area_v2") == "Set the fabric area manually"
    if not manual:
        return None
    st.caption("Set a box around the fabric opening. Only the oval inside that box is used.")
    columns = st.columns(4)
    return tuple(column.number_input(label, minimum, value=default, key=f"{key}_{label}")
                 for column, label, minimum, default in
                 zip(columns, ("X", "Y", "Width", "Height"), (0, 0, 1, 1), (0, 0, 640, 480)))


def test_settings(key: str):
    model = active_model()
    if not model.is_file():
        st.warning("The droplet model is missing. Train the model or restore the project weights.")
        return None
    with st.expander("Detection settings"):
        every_frame = st.checkbox("Analyze every frame", key=f"{key}_every_frame",
                                  help="Use this for brief droplets. Analysis may take longer than playback.")
        confidence = st.slider("Minimum confidence", .05, .95, .25, .05, key=f"{key}_conf")
        hits = st.number_input("Frame observations to confirm a leak", 1, 30, 3, key=f"{key}_hits")
        distance = st.number_input("Position matching distance (px)", 1, 100, 10, key=f"{key}_distance")
        roi = fabric_area_setting(key)
        if roi is None:
            st.caption("The cyan outline shows the allowed fabric area. Detection pauses if the white rim cannot be located.")
        surface_water = st.checkbox("Detect spreading water and pools", value=True,
                                    key=f"{key}_surface_water")
        st.caption("Start with dry fabric. Spreading water is reported separately from individual droplets; "
                   "the amber line in its snapshot marks an observed water boundary, not the leak origin.")
        new_only = st.checkbox("Ignore marks already present in the opening frame", value=True,
                               key=f"{key}_new_only",
                               help="Turn this off if droplets are already visible at the start.")
    return dict(model_path=model, confidence_threshold=confidence,
                confirmation_hits=hits, match_distance_px=distance, roi=roi, roi_shape="ellipse",
                analysis_fps=None if every_frame else FAST_ANALYSIS_FPS,
                auto_fabric_area=roi is None, require_new_appearance=new_only,
                detect_surface_water=surface_water)


def show_preview(preview: Preview, waiting: str) -> None:
    frame = preview.latest
    if frame is None:
        st.info(waiting)
        return
    with st.container(border=True):
        elapsed, count = st.columns(2)
        elapsed.metric("Elapsed test time", f"{frame.elapsed_s:.1f} s")
        count.metric("Confirmed droplets", f"{frame.leak_count} / 3")
        if frame.water_detected:
            st.warning("Visible spreading water detected inside the fabric area.")
        st.image(frame.jpeg, width="stretch")


def validation(result, key: str) -> None:
    with st.expander("Compare with manually verified leaks"):
        st.write("Add a CSV with x_px, y_px and time_s columns. Times are relative to test start. "
                 "This comparison measures individual droplets only. A file with headers only represents "
                 "a run with no countable droplets; spreading-water warnings are separate.")
        st.download_button("Get CSV template", "x_px,y_px,time_s\n", "reference.csv",
                           mime="text/csv", key=f"{key}_template")
        uploaded = st.file_uploader("Reference leak CSV", type="csv", key=f"{key}_reference")
        limits = st.columns(2)
        distance = limits[0].number_input("Position tolerance (px)", 1, 1000, 20, key=f"{key}_px")
        timing = limits[1].number_input("Timing tolerance (s)", .01, 60.0, .5, key=f"{key}_time")
        if not uploaded:
            return
        try:
            reader = csv.DictReader(io.StringIO(uploaded.getvalue().decode("utf-8-sig")))
            if not {"x_px", "y_px", "time_s"} <= set(reader.fieldnames or []):
                raise ValueError("Expected x_px, y_px and time_s columns.")
            references = []
            for row in reader:
                values = tuple(float(row[field]) for field in ("x_px", "y_px", "time_s"))
                if any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError("Reference values must be finite and non-negative.")
                references.append(values)
                if len(references) > 3:
                    raise ValueError("Include only the first three distinct reference leaks.")
        except (ValueError, TypeError, UnicodeError, csv.Error) as exc:
            st.error(str(exc))
            return
        # Maximize the number of one-to-one position matches, then minimize distance.
        best = (0, math.inf, ())
        for indexes in itertools.product(range(-1, len(references)), repeat=len(result.leaks)):
            used = [idx for idx in indexes if idx >= 0]
            if len(set(used)) != len(used):
                continue
            pairs = [(leak, references[idx]) for leak, idx in zip(result.leaks, indexes) if idx >= 0]
            gaps = [math.hypot(leak.x_px - x, leak.y_px - y) for leak, (x, y, _) in pairs]
            if all(gap <= distance for gap in gaps) and (-len(pairs), sum(gaps)) < best[:2]:
                best = (-len(pairs), sum(gaps), pairs)
        pairs = best[2]
        errors = [abs(leak.detection_time_s - ref[2]) for leak, ref in pairs]
        counts = {"Matched": len(pairs), "False positives": len(result.leaks) - len(pairs),
                  "Missed": len(references) - len(pairs),
                  "Outside timing tolerance": sum(error > timing for error in errors)}
        for column, (name, count) in zip(st.columns(4), counts.items()):
            column.metric(name, count)
        if errors:
            st.caption(f"Mean absolute timing error: {sum(errors) / len(errors):.3f} s")
        precision = f"{len(pairs) / len(result.leaks):.0%}" if result.leaks else "N/A"
        miss_rate = f"{counts['Missed'] / len(references):.0%}" if references else "N/A"
        st.caption(f"Precision: {precision} · Missed-leak rate: {miss_rate}")
        st.altair_chart(alt.Chart(alt.Data(values=[
            {"Outcome": name, "Count": count} for name, count in counts.items()
        ])).mark_bar(color=TEAL).encode(x="Count:Q", y=alt.Y("Outcome:N", sort=None))
            .properties(height=150), width="stretch")
        st.caption("Matching uses position tolerance. Timing error is measured separately for matched leaks.")


def show_results(result, key: str) -> None:
    st.subheader("Test results")
    for error in result.errors:
        st.error(error)
    for warning in result.warnings:
        st.warning(warning)
    total, first, status = st.columns(3)
    total.metric("Confirmed droplets", f"{len(result.leaks)} / 3")
    times = [leak.detection_time_s for leak in result.leaks]
    if result.surface_water is not None:
        times.append(result.surface_water.detection_time_s)
    first_time = min(times, default=None)
    first.metric("First water observation", f"{first_time:.2f} s" if first_time is not None else "—")
    status.metric("Test status", result.status.capitalize())
    if result.surface_water is not None:
        water = result.surface_water
        st.warning(f"Visible water leakage detected at {water.detection_time_s:.2f} s "
                   f"and confirmed at {water.confirmation_time_s:.2f} s. "
                   "Spreading water does not add to the individual droplet count.")
        if water.image_filename:
            st.image(water.image_filename, caption="Observed water boundary (amber). The leak origin is not determined.",
                     width="stretch")
    if not result.leaks and result.surface_water is None and result.status != "error":
        st.info("No water leakage was confirmed by the analyzer. This does not establish that the fabric is dry.")
    if result.final_image_filename:
        with st.container(border=True):
            st.markdown("#### Final detection map")
            st.image(result.final_image_filename, width="stretch")
    if result.leaks:
        events = [{
            "Leak": event.droplet_number, "Detected (s)": round(event.detection_time_s, 3),
            "Confirmed (s)": round(event.confirmation_time_s, 3), "X (px)": round(event.x_px, 1),
            "Y (px)": round(event.y_px, 1), "Confidence": round(event.confidence, 3),
        } for event in result.leaks]
        st.dataframe(events, hide_index=True, width="stretch")
        timeline, positions = st.columns(2)
        points = alt.Chart(alt.Data(values=events)).mark_circle(color=TEAL, size=115)
        tooltip = ["Leak:O", "Detected (s):Q", "X (px):Q", "Y (px):Q", "Confidence:Q"]
        timeline.altair_chart(points.encode(
            x=alt.X("Detected (s):Q", scale=alt.Scale(zero=True)),
            y=alt.Y("Leak:O", sort="descending"), tooltip=tooltip,
        ).properties(title="Leak timeline", height=190), width="stretch")
        x_domain, y_domain = alt.Scale(zero=True), alt.Scale(reverse=True, zero=True)
        if result.final_image_filename:
            with Image.open(result.final_image_filename) as image:
                x_domain = alt.Scale(domain=[0, image.width])
                y_domain = alt.Scale(domain=[image.height, 0])
        positions.altair_chart(points.encode(
            x=alt.X("X (px):Q", scale=x_domain), y=alt.Y("Y (px):Q", scale=y_domain),
            tooltip=tooltip,
        ).properties(title="Leak positions · camera pixels", height=190), width="stretch")
        with st.expander("Images at detection"):
            for col, event in zip(st.columns(3), result.leaks):
                if event.image_filename:
                    col.image(event.image_filename, caption=f"Leak {event.droplet_number}", width="stretch")
    if result.results_filename:
        st.download_button("Download results JSON", result.model_dump_json(indent=2),
                           "results.json", mime="application/json", key=f"{key}_json")
    if result.video_filename:
        st.caption("Recording saved on this computer")
        st.code(result.video_filename, language=None)
    if result.status != "error":
        validation(result, key)


def recorded_page() -> None:
    st.header("Analyze a recording")
    upload = st.file_uploader("Test video", type=VIDEO_TYPES, key="replay_upload",
                              help="H.264 MP4 and WebM play directly in the browser.")
    if upload is not None:
        preview_format = PREVIEW_FORMATS.get(Path(upload.name).suffix.lower().lstrip("."))
        if preview_format:
            st.video(upload, format=preview_format)
        else:
            st.caption("This file can be analyzed. Use H.264 MP4 or WebM for browser playback.")
    settings = test_settings("replay")
    start = st.number_input("Test start in the recording (seconds)", min_value=0.0, value=0.0,
                            help="Leave at zero if the recording begins when the test starts.")
    if st.button("Analyze video", type="primary", disabled=upload is None or settings is None or busy()):
        try:
            folder = RUNS / f"recorded-{uuid4().hex[:12]}"
            source = folder / ("source" + Path(upload.name).suffix.lower())
            save_upload(upload, source)
            try:
                frame_count, fps = video_info(source)
                duration = frame_count / fps - start
            except (ValueError, cv2.error):
                duration = None  # Some playable recordings have no reported frame count.
            if duration is not None and duration <= 0:
                raise ValueError("Test start must be before the end of the video.")
            progress = ReplayProgress()
            st.session_state.replay_name = upload.name
            st.session_state.replay_upload_id = upload.file_id
            st.session_state.replay_progress = progress
            st.session_state.replay_duration = duration
            st.session_state.replay = workers().submit(
                analyze_video, source, outputs=OutputPaths.in_directory(folder),
                settings=VideoSettings(**settings), test_start_s=start, on_progress=progress.publish,
                use_source_recording=True,
            )
            st.rerun()
        except (OSError, ValueError, cv2.error) as exc:
            st.error(str(exc))
    job = st.session_state.get("replay")
    if job is not None:
        if job.done() and upload is not None and upload.file_id != st.session_state.get("replay_upload_id"):
            return
        st.caption(f"Recording: {st.session_state.get('replay_name', 'Uploaded video')}")
        if job.done():
            show_results(job.result(), "replay")
        else:
            @st.fragment(run_every=.5)
            def analysis_status():
                if job.done():
                    st.rerun()
                elapsed = st.session_state.replay_progress.seconds
                if elapsed is None:
                    st.info("Loading the detector… You can play or seek the video now.")
                else:
                    duration = st.session_state.replay_duration
                    if duration is None:
                        st.caption(f"Analyzed through {elapsed:.1f} s of video")
                    else:
                        st.progress(min(1.0, elapsed / duration),
                                    text=f"Analyzed {elapsed:.1f} of {duration:.1f} s of video")
            analysis_status()


def live_page() -> None:
    st.header("Monitor a live camera")
    st.write("Connect a USB camera to the computer running this dashboard. "
             "Start monitoring before the first droplet appears so the camera has time to open.")
    settings = test_settings("live")
    controls = st.columns(4)
    camera = controls[0].number_input("Camera ID", 0, 20, 0)
    width = controls[1].number_input("Frame width", 160, 7680, 1280)
    height = controls[2].number_input("Frame height", 120, 4320, 720)
    fps = controls[3].number_input("Requested FPS", 1, 120, 30)
    if st.button("Start camera test", type="primary", disabled=settings is None or busy()):
        preview = Preview()
        st.session_state.live_preview = preview
        st.session_state.live_stop = False
        st.session_state.live = start_test(
            camera, outputs=OutputPaths.in_directory(RUNS / f"live-{uuid4().hex[:12]}"),
            settings=VideoSettings(**settings, resolution=(width, height), frame_rate=fps),
            on_preview=preview.publish,
        )
        st.rerun()
    session = st.session_state.get("live")
    if session is not None:
        if session.done:
            show_results(session.stop_test(), "live")
        else:
            @st.fragment(run_every=.1)
            def monitor():
                if session.done:
                    st.rerun()
                if st.button("Stop test and save results", disabled=st.session_state.live_stop):
                    session.request_stop()
                    st.session_state.live_stop = True
                if st.session_state.live_stop:
                    st.info("Finalizing the recording and results…")
                show_preview(st.session_state.live_preview, "Opening the camera and loading the detection model…")
            monitor()


def video_labelling(root: Path) -> str | None:
    st.subheader("Label frames from a video")
    st.write("Draw a box around each visible droplet and select Save boxes. For a dry frame, select Save as dry.")
    upload = st.file_uploader("Video to label", type=VIDEO_TYPES, key="training_upload")
    if upload is None:
        return
    clip = st.session_state.get("general_annotation_clip")
    if clip is None or clip[0] != upload.file_id:
        clip_id = hashlib.sha256(upload.getbuffer()).hexdigest()
        source = root / "videos" / (clip_id + Path(upload.name).suffix.lower())
        try:
            if not source.exists():
                save_upload(upload, source)
            frame_count, fps = video_info(source)
        except (OSError, ValueError, cv2.error) as exc:
            st.error(str(exc))
            return
        clip = (upload.file_id, clip_id, source, frame_count, fps)
        st.session_state.general_annotation_clip = clip

    _, clip_id, source, frame_count, fps = clip
    nav_key = f"general_frame_{clip_id}"
    time_key = f"general_seconds_{clip_id}"
    end_time = round((frame_count - 1) / fps, 2)
    st.session_state.setdefault(nav_key, 1)
    st.session_state.setdefault(time_key, round((st.session_state[nav_key] - 1) / fps, 2))

    def set_frame(position: int) -> None:
        position = max(1, min(frame_count, position))
        st.session_state[nav_key] = position
        st.session_state[time_key] = round((position - 1) / fps, 2)

    def move(delta: int) -> None:
        set_frame(st.session_state[nav_key] + delta)

    def jump_time() -> None:
        set_frame(1 + round(st.session_state[time_key] * fps))

    step = max(1, round(5 * fps))
    at_start, at_end = st.session_state[nav_key] <= 1, st.session_state[nav_key] >= frame_count
    time_nav = st.columns([1, 2, 1])
    time_nav[0].button("Back 5 s", on_click=move, args=(-step,), disabled=at_start, width="stretch")
    time_nav[1].number_input("Jump to time (seconds)", min_value=0.0, max_value=end_time,
                             step=.25, format="%.2f", key=time_key, on_change=jump_time,
                             disabled=frame_count == 1)
    time_nav[2].button("Forward 5 s", on_click=move, args=(step,), disabled=at_end, width="stretch")

    frame_nav = st.columns([1, 2, 1])
    frame_nav[0].button("Previous frame", on_click=move, args=(-1,), disabled=at_start, width="stretch")
    position = frame_nav[1].number_input("Frame number", min_value=1, max_value=frame_count,
                                         step=1, key=nav_key, on_change=move, args=(0,),
                                         disabled=frame_count == 1)
    frame_nav[2].button("Next frame", on_click=move, args=(1,), disabled=at_end, width="stretch")

    frame_id = f"general/{clip_id}/{position}"
    path = label_path(root, "train", clip_id, position - 1)
    try:
        frame, timestamp = read_frame(source, position - 1)
        preview = frame
        if frame.shape[1] > 1200:
            preview = cv2.resize(frame, (1200, max(1, round(1200 * frame.shape[0] / frame.shape[1]))))
        ok, image = cv2.imencode(".jpg", preview)
        if not ok:
            raise OSError("Could not open the frame in the editor.")
        boxes = load_boxes(path)
        st.caption(f"Frame {position:,} of {frame_count:,} · {timestamp:.2f} s into the video")
        locked = busy()
        reply = box_editor(image="data:image/jpeg;base64," + base64.b64encode(image).decode("ascii"),
                           frame_id=frame_id, boxes=boxes, locked=locked, key="general_annotation_editor")
        if (isinstance(reply, dict) and reply.get("frameId") == frame_id
                and reply.get("id") != st.session_state.get("general_annotation_saved_event") and not locked):
            boxes, dry = reply.get("boxes"), reply.get("dry")
            if not isinstance(boxes, list) or type(dry) is not bool:
                raise ValueError("Invalid annotation. Try saving the frame again.")
            lock = training_lock()
            if not lock.acquire(blocking=False):
                st.info("Wait for training to finish before saving more labels.")
                return clip_id
            try:
                save_frame(root, "train", clip_id, position - 1, frame, boxes, dry=dry)
            finally:
                lock.release()
            st.session_state.general_annotation_saved_event = reply.get("id")
            st.success("Saved as a dry example." if dry else f"Saved {len(boxes)} droplet boxes.")
        elif path.is_file():
            st.caption(f"Saved: {len(boxes)} droplet boxes" if boxes else "Saved as dry")
    except (OSError, ValueError, cv2.error) as exc:
        st.error(str(exc))
    return clip_id


def training_page() -> None:
    st.header("Train the model")
    source = st.radio("Training data", ["Label video frames", "Import labelled CVAT ZIP"], horizontal=True)
    in_app = source == "Label video frames"
    archive = None
    focus_clip = None
    manual_roi = None
    if in_app:
        focus_clip = video_labelling(ANNOTATIONS)
        pattern = f"{focus_clip}_*.txt" if focus_clip else "*.txt"
        labels = list((ANNOTATIONS / "labels" / "train").glob(pattern))
        positive = sum(path.stat().st_size > 0 for path in labels)
        st.caption(f"{len(labels)} saved frames · {positive} with droplets")
        ready = positive >= 2
        if not ready:
            st.info("Save droplet boxes on at least two different frames to start training.")
        elif focus_clip:
            st.caption("Training uses the fabric inside the rim and holds back labeled frames from this recording to check the result.")
    else:
        archive = cvat_upload()
        ready = bool(archive)
    with st.expander("Training settings"):
        epochs = st.number_input("Training passes", 1, 150, 24)
        if focus_clip:
            manual_roi = fabric_area_setting("training")
            if manual_roi:
                st.caption("Use the same oval under Detection settings when you replay this video.")
    if st.button("Train the model", type="primary", disabled=not ready or busy()):
        base_model = active_model()
        if not base_model.is_file():
            base_model = ROOT / "yolo11n.pt"
        lock = training_lock()
        if not base_model.is_file():
            st.error("The starting model weights are missing.")
        elif lock.acquire(blocking=False):
            folder = RUNS / f"training-{uuid4().hex[:8]}"
            try:
                if in_app:
                    future = workers().submit(train_annotated_model, folder, ACTIVE_MODEL,
                                              base_model, epochs, ANNOTATIONS,
                                              focus_clip=focus_clip, manual_roi=manual_roi,
                                              analysis_fps=FAST_ANALYSIS_FPS)
                else:
                    save_upload(archive, folder / "dataset.zip")
                    future = workers().submit(train_profile, folder, ACTIVE_MODEL, base_model, epochs)
            except Exception as exc:
                lock.release()
                st.error(str(exc))
            else:
                future.add_done_callback(lambda _: lock.release())
                st.session_state.training = future
                st.session_state.training_folder = folder
                st.rerun()
        else:
            st.info("Training is already in progress.")
    show_training_status()


def cvat_upload():
    with st.container(border=True):
        st.markdown("#### Prepare the labelled examples")
        st.write("In CVAT, mark real droplets as water_droplet (class 0). Export as Ultralytics YOLO Detection "
                 "with source images. Upload one ZIP containing images/train, images/val, labels/train and labels/val. "
                 "Keep different test recordings in the two splits. Missing labels are treated as no-droplet frames.")
    return st.file_uploader("Labelled CVAT dataset ZIP", type="zip")


def show_training_status() -> None:
    job = st.session_state.get("training")
    if job is not None:
        if job.done():
            try:
                job.result()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            else:
                st.success("Training complete. Recorded video and Live camera now use the updated detector.")
                folder = st.session_state.get("training_folder")
                if folder is not None and (folder / "training_check.json").is_file():
                    check = json.loads((folder / "training_check.json").read_text(encoding="utf-8"))
                    validation = check["validation"]
                    st.caption(f"Validation matched {validation['matched']} of {validation['labels']} held-back droplet boxes.")
                    replay = folder / "replay-validation" / "results.json"
                    if replay.is_file():
                        count = len(json.loads(replay.read_text(encoding="utf-8"))["leaks"])
                        st.caption(f"Replay confirmed {count} separate leak positions, including a position you marked.")
                    skipped = folder / "skipped_frames.json"
                    if skipped.is_file():
                        count = len(json.loads(skipped.read_text(encoding="utf-8")))
                        st.warning(f"Skipped {count} saved frames with an unavailable fabric area or boxes outside the usable oval.")
        else:
            @st.fragment(run_every=2)
            def training_status():
                if job.done():
                    st.rerun()
                folder = st.session_state.get("training_folder")
                if folder is not None and (folder / "replay-validation").is_dir():
                    st.info("Replaying the labeled recording to check detections before switching models.")
                else:
                    st.info("Preparing the dataset and training the model. Keep this dashboard open to check the result.")
            training_status()


def main() -> None:
    st.set_page_config(page_title="Hydrostatic Leak Detection", layout="wide")
    st.markdown("""
        <style>
        .block-container {max-width: 1120px; padding-top: 2.7rem; padding-bottom: 4rem}
        [data-testid="stMetric"] {background:#eff7f6; border-radius:12px; padding:14px 17px}
        [data-testid="stMetricValue"] {color:#075d58}
        h1, h2, h3 {letter-spacing:-0.025em}
        </style>
    """, unsafe_allow_html=True)
    st.title("Hydrostatic Leak Detection")
    st.caption("Water droplet detection for fabric pressure tests")
    pages = {"Recorded video": recorded_page, "Live camera": live_page, "Train the Model": training_page}
    current = st.radio("Workspace", list(pages), horizontal=True, label_visibility="collapsed")
    st.divider()
    pages[current]()


if __name__ == "__main__":
    main()
