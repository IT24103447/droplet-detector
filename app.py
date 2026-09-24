"""Simple local workflow for labelling and detecting fabric water droplets."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from droplet_detector.dataset import dataset_counts, save_labelled_image
from droplet_detector.single_image import ModelNotReadyError, detect_droplets_in_single_image

MODEL_PATH = Path("models/droplet-seg.pt")
UPLOAD_DIR = Path("data/uploads/single_image")


def draw_detections(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw a labelled green circle around each model detection."""
    result = image_bgr.copy()
    for detection in detections:
        radius = int(round(detection.radius_px))
        center = (int(round(detection.x_px)), int(round(detection.y_px)))
        cv2.circle(result, center, radius, (0, 220, 0), 2)
        cv2.putText(
            result,
            f"#{detection.droplet_number} {detection.confidence:.0%}",
            (center[0], max(20, center[1] - radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return result


def canvas_image(uploaded_file) -> tuple[Image.Image, int, int]:
    """Resize a photo for the drawing canvas while preserving its aspect ratio."""
    image = Image.open(uploaded_file).convert("RGB")
    width = min(image.width, 900)
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height)), width, height


def main() -> None:
    st.set_page_config(page_title="Water Droplet Detector", page_icon="💧", layout="wide")
    st.title("💧 Water Droplet Detector")
    st.caption("Label examples → train one local model → detect water from one photo.")

    sensitivity = st.sidebar.slider(
        "Detection sensitivity", 0.1, 1.0, 0.5, 0.05,
        help="Lower values find more possible droplets; higher values are stricter.",
    )
    counts = dataset_counts()
    st.sidebar.markdown("### Training progress")
    st.sidebar.write(f"Training images: **{counts['train']}**")
    st.sidebar.write(f"Validation images: **{counts['val']}**")

    label_tab, train_tab, detect_tab = st.tabs([
        "1. Label training photos",
        "2. Train the model",
        "3. Detect one photo",
    ])

    with label_tab:
        st.subheader("Label photos so the model can learn")
        st.write(
            "Upload one fabric photo. Draw one rectangle around each real water droplet. "
            "For a dry photo, draw no rectangles and save it as a no-water example."
        )
        split_label = st.radio(
            "Where should this example go?", ["Training set", "Validation set"], horizontal=True,
            help="Do not use near-identical video frames in both sets.",
        )
        split = "train" if split_label == "Training set" else "val"
        source = st.file_uploader(
            "Choose one fabric photo", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="label_photo"
        )
        if source is not None:
            display_image, canvas_width, canvas_height = canvas_image(source)
            st.write("Draw rectangles around droplets. Use the selection tool to move or resize a rectangle.")
            canvas = st_canvas(
                fill_color="rgba(0, 180, 255, 0.20)",
                stroke_width=2,
                stroke_color="#00b4ff",
                background_image=display_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key=f"canvas_{source.name}",
            )
            objects = []
            if canvas.json_data:
                objects = [item for item in canvas.json_data.get("objects", []) if item.get("type") == "rect"]
            st.info(f"Rectangles drawn: {len(objects)}")
            if st.button("Save labelled example", type="primary"):
                image_path, _, droplet_count = save_labelled_image(
                    source.name, source.getvalue(), split, objects, canvas_width, canvas_height
                )
                if droplet_count:
                    st.success(f"Saved {image_path.name} with {droplet_count} labelled droplet(s).")
                else:
                    st.success(f"Saved {image_path.name} as a no-water example.")
                st.rerun()

    with train_tab:
        st.subheader("Train your local model")
        st.write(
            "First label fabric photos in Step 1. Include dry fabric, water droplets, "
            "reflections, folds, and different fabric types."
        )
        col_train, col_val = st.columns(2)
        col_train.metric("Training photos", counts["train"])
        col_val.metric("Validation photos", counts["val"])
        if counts["train"] == 0 or counts["val"] == 0:
            st.warning("Add labelled training and validation photos before training.")
        else:
            st.success("Your dataset is ready to train.")
        st.code('pip install -e ".[ml]"\npython scripts/train_detection.py')
        st.caption(
            "After training, copy the generated best.pt file to models/droplet-seg.pt. "
            "Then Step 3 will detect droplets from one image."
        )

    with detect_tab:
        st.subheader("Detect water droplets in one photo")
        st.write("Upload one photo. No dry reference is required.")
        photo = st.file_uploader(
            "Choose a fabric photo", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="detect_photo"
        )
        if photo is not None:
            st.image(photo, caption=photo.name, width=600)
            if st.button("Detect water droplets", type="primary"):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                image_path = UPLOAD_DIR / Path(photo.name).name
                image_path.write_bytes(photo.getvalue())
                try:
                    detections = detect_droplets_in_single_image(image_path, MODEL_PATH, sensitivity)
                    image_bgr = cv2.imread(str(image_path))
                    annotated = draw_detections(image_bgr, detections)
                    st.metric("Water droplets detected", len(detections))
                    st.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Detected water droplets", use_container_width=True,
                    )
                    if not detections:
                        st.success("No water droplets detected.")
                except ModelNotReadyError as exc:
                    st.warning("Train the local model first in Step 2.")
                    st.info(str(exc))
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")


if __name__ == "__main__":
    main()
