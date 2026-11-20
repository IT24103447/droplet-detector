"""Lightweight operator dashboard for the hydrotester library."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import itertools
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import altair as alt
import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from droplet_detector import OutputPaths, VideoSettings, analyze_video, start_test
from droplet_detector.fabric_training import train_annotated_profile, train_profile
from droplet_detector.video import VideoPreview
from droplet_detector.video_annotation import label_path, load_boxes, read_frame, save_frame

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "droplet-seg.pt"
PROFILES = ROOT / "models" / "fabrics"
RUNS = ROOT / "data" / "results" / "dashboard"
ANNOTATIONS = ROOT / "data" / "fabric_annotations"
TEAL = "#087F79"
box_editor = components.declare_component("droplet_boxes", path=str(ROOT / "ui" / "box_editor"))


class Preview:
    """A worker replaces the latest preview; only Streamlit renders it."""

    latest: VideoPreview | None = None

    def publish(self, frame: VideoPreview) -> None:
        self.latest = frame


@st.cache_resource
def workers():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="hydrotester")


def save_upload(upload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    upload.seek(0)
    with path.open("xb") as target:
        shutil.copyfileobj(upload, target)


def busy() -> bool:
    return (any(job is not None and not job.done()
                for job in (st.session_state.get("replay"), st.session_state.get("training")))
            or (st.session_state.get("live") is not None and not st.session_state.live.done))


def test_settings(key: str):
    profiles = {"General model": DEFAULT_MODEL}
    profiles.update((p.stem.replace("_", " "), p) for p in sorted(PROFILES.glob("*.pt")))
    available = {name: path for name, path in profiles.items() if path.is_file()}
    if not available:
        st.warning("No trained model is available. Add your trained weights or create a fabric model.")
        return None
    model = st.selectbox("Detection model", list(available), key=f"{key}_model")
    with st.expander("Detection settings"):
        confidence = st.slider("Minimum confidence", .05, .95, .25, .05, key=f"{key}_conf")
        hits = st.number_input("Frame observations to confirm a leak", 1, 30, 3, key=f"{key}_hits")
        distance = st.number_input("Position matching distance (px)", 1, 100, 10, key=f"{key}_distance")
        roi = None
        if st.checkbox("Limit detection to a fabric area", key=f"{key}_roi"):
            columns = st.columns(4)
            roi = tuple(column.number_input(label, minimum, value=default, key=f"{key}_{label}")
                        for column, label, minimum, default in
                        zip(columns, ("X", "Y", "Width", "Height"), (0, 0, 1, 1), (0, 0, 640, 480)))
    return dict(model_path=available[model], confidence_threshold=confidence,
                confirmation_hits=hits, match_distance_px=distance, roi=roi)


def show_preview(preview: Preview, waiting: str) -> None:
    frame = preview.latest
    if frame is None:
        st.info(waiting)
        return
    with st.container(border=True):
        elapsed, count = st.columns(2)
        elapsed.metric("Elapsed test time", f"{frame.elapsed_s:.1f} s")
        count.metric("Confirmed leaks", f"{frame.leak_count} / 3")
        st.image(frame.jpeg, width="stretch")


def validation(result, key: str) -> None:
    with st.expander("Compare with manually verified leaks"):
        st.write("Add a CSV with x_px, y_px and time_s columns. Times are relative to test start. "
                 "A file with headers only represents a run with no leaks.")
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
    total.metric("Confirmed leaks", f"{len(result.leaks)} / 3")
    first_time = min((leak.detection_time_s for leak in result.leaks), default=None)
    first.metric("First leak", f"{first_time:.2f} s" if first_time is not None else "—")
    status.metric("Test status", result.status.capitalize())
    if not result.leaks and result.status != "error":
        st.info("No leak was confirmed in this run.")
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
    st.write("Upload a company test video to locate and time the first three distinct leaks.")
    settings = test_settings("replay")
    upload = st.file_uploader("Recorded test video", type=["mp4", "mov", "avi", "mkv", "webm"])
    start = st.number_input("Test start in the recording (seconds)", min_value=0.0, value=0.0,
                            help="Leave at zero if the recording begins when the test starts.")
    if st.button("Analyze video", type="primary", disabled=not upload or settings is None or busy()):
        try:
            folder = RUNS / f"recorded-{uuid4().hex[:12]}"
            source = folder / ("source" + Path(upload.name).suffix.lower())
            save_upload(upload, source)
            preview = Preview()
            st.session_state.replay_name = upload.name
            st.session_state.replay_preview = preview
            st.session_state.replay = workers().submit(
                analyze_video, source, outputs=OutputPaths.in_directory(folder),
                settings=VideoSettings(**settings), test_start_s=start, on_preview=preview.publish,
            )
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    job = st.session_state.get("replay")
    if job is not None:
        st.caption(f"Recording: {st.session_state.get('replay_name', 'Uploaded video')}")
        if job.done():
            show_results(job.result(), "replay")
        else:
            @st.fragment(run_every=.5)
            def progress():
                if job.done():
                    st.rerun()
                show_preview(st.session_state.replay_preview, "Opening video and loading the detection model…")
            progress()


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
            @st.fragment(run_every=.5)
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


def video_labelling(root: Path) -> bool:
    st.subheader("Label frames from a video")
    st.write("Choose a split for this entire recording. Draw a tight box around every visible droplet "
             "on each frame you save. Save a few clearly dry frames as well.")
    split = st.radio("Recording split", ["Training", "Validation"], horizontal=True)
    split = "train" if split == "Training" else "val"
    upload = st.file_uploader("Video to label", type=["mp4", "mov", "avi", "mkv", "webm"])
    if st.button("Open in frame editor", disabled=not upload):
        clip_id = hashlib.sha256(upload.getbuffer()).hexdigest()
        source = root / "videos" / (clip_id + Path(upload.name).suffix.lower())
        try:
            if not source.exists():
                save_upload(upload, source)
            capture = cv2.VideoCapture(str(source))
            try:
                raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if not capture.isOpened() or not math.isfinite(raw_count) or raw_count < 1:
                    raise ValueError("Could not read the frame count. Try saving the recording as an MP4.")
            finally:
                capture.release()
            st.session_state.annotation_clip = (str(root), clip_id, source, int(raw_count), upload.name)
        except (OSError, ValueError) as exc:
            st.error(str(exc))

    clip = st.session_state.get("annotation_clip")
    if clip and clip[0] == str(root):
        _, clip_id, source, frame_count, name = clip
        st.caption(f"Editing {name}")
        nav_key = f"annotation_frame_{root.name}_{clip_id}"
        st.session_state.setdefault(nav_key, 1)
        nav = st.columns([1, 2, 1])

        def move(delta: int) -> None:
            st.session_state[nav_key] += delta

        nav[0].button("Previous frame", on_click=move, args=(-1,), disabled=st.session_state[nav_key] <= 1)
        position = nav[1].number_input("Frame number", min_value=1, max_value=frame_count,
                                       step=1, key=nav_key)
        nav[2].button("Next frame", on_click=move, args=(1,), disabled=st.session_state[nav_key] >= frame_count)
        path = label_path(root, split, clip_id, position - 1)
        try:
            frame, timestamp = read_frame(source, position - 1)
            preview = frame
            if frame.shape[1] > 1200:
                preview = cv2.resize(frame, (1200, max(1, round(1200 * frame.shape[0] / frame.shape[1]))))
            ok, image = cv2.imencode(".jpg", preview)
            if not ok:
                raise OSError("Could not open the frame in the editor.")
            boxes = load_boxes(path)
            st.caption(f"Frame {position:,} of {frame_count:,} · {timestamp:.2f} s into the clip")
            other = "val" if split == "train" else "train"
            wrong_split = any((root / "labels" / other).glob(f"{clip_id}_*.txt"))
            if wrong_split:
                st.warning("This recording already has labels in the other split. Select that split to edit them.")
            frame_id = f"{root.name}/{split}/{clip_id}/{position}"
            reply = box_editor(image="data:image/jpeg;base64," + base64.b64encode(image).decode("ascii"),
                               frame_id=frame_id, boxes=boxes, locked=wrong_split or busy(),
                               key="annotation_editor")
            if (isinstance(reply, dict) and reply.get("frameId") == frame_id
                    and reply.get("id") != st.session_state.get("annotation_saved_event")
                    and not wrong_split and not busy()):
                st.session_state.annotation_saved_event = reply.get("id")
                boxes = reply.get("boxes")
                dry = reply.get("dry")
                if not isinstance(boxes, list) or type(dry) is not bool:
                    raise ValueError("Invalid annotation. Try saving the frame again.")
                save_frame(root, split, clip_id, position - 1, frame, boxes, dry=dry)
                st.success("Saved as a dry example." if dry else f"Saved {len(boxes)} droplet boxes.")
            elif path.is_file():
                st.caption(f"Saved: {len(boxes)} droplet boxes" if boxes else "Saved as dry")
        except (OSError, ValueError, cv2.error) as exc:
            st.error(str(exc))
    counts = {group: list((root / "labels" / group).glob("*.txt")) for group in ("train", "val")}
    columns = st.columns(2)
    for column, (group, files) in zip(columns, counts.items()):
        column.metric("Training frames" if group == "train" else "Validation frames", len(files))
    return all(any(path.stat().st_size for path in files) for files in counts.values())


def fabric_page() -> None:
    st.header("Create a fabric model")
    st.write("Fine-tune the general model for a fabric that needs more accurate detection.")
    name = st.text_input("Fabric profile name", placeholder="e.g. Woven nylon", max_chars=60)
    slug = re.sub("[^A-Za-z0-9]+", "_", name).strip("_")
    source = st.radio("Training data", ["Label video frames", "Import labelled CVAT ZIP"], horizontal=True)
    in_app = source == "Label video frames"
    archive = None
    annotations = None
    ready = False
    if in_app:
        if slug:
            # Hash the profile name so different names never share a dataset accidentally.
            profile_key = hashlib.sha256(name.strip().casefold().encode()).hexdigest()[:16]
            annotations = ANNOTATIONS / f"{slug.lower()}_{profile_key}"
            ready = video_labelling(annotations)
            if not ready:
                st.info("Save at least one droplet-labelled frame in each split to enable training. "
                        "Use distinct recordings for training and validation.")
        else:
            st.info("Enter a fabric profile name to begin labelling.")
    else:
        archive = cvat_upload()
        ready = bool(archive)
    epochs = st.number_input("Training epochs", 1, 150, 30)
    if st.button("Train fabric model", type="primary", disabled=not slug or not ready or busy()):
        base_model = DEFAULT_MODEL if DEFAULT_MODEL.is_file() else ROOT / "yolo11n.pt"
        if not base_model.is_file():
            st.error("Place the general droplet model or YOLO11n weights in the project first.")
        else:
            job_id = uuid4().hex[:8]
            folder = RUNS / f"training-{job_id}"
            destination = PROFILES / f"{slug}_{job_id}.pt"
            try:
                if in_app:
                    future = workers().submit(train_annotated_profile, folder, destination,
                                              base_model, epochs, annotations)
                else:
                    save_upload(archive, folder / "dataset.zip")
                    future = workers().submit(train_profile, folder, destination, base_model, epochs)
                st.session_state.training = future
                st.rerun()
            except OSError as exc:
                st.error(str(exc))
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
                model = job.result()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            else:
                st.success(f"{model.stem.replace('_', ' ')} is available in the detection model menu.")
        else:
            @st.fragment(run_every=2)
            def training_status():
                if job.done():
                    st.rerun()
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
    pages = {"Recorded video": recorded_page, "Live camera": live_page, "Fabric models": fabric_page}
    current = st.radio("Workspace", list(pages), horizontal=True, label_visibility="collapsed")
    st.divider()
    pages[current]()


if __name__ == "__main__":
    main()
