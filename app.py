"""Streamlit Web UI for Droplet Detector.

Allows uploading dry reference photos (with fabric type selection),
uploading raw test photos with droplets (organized by fabric and optional test run),
and interactively running detection and viewing visual results.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from droplet_detector.config import DropletDetectorConfig
from droplet_detector.pipeline import detect_droplets_in_image
from droplet_detector.preprocessing import diff_against_dry_reference

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
DRY_REF_DIR = DATA_ROOT / "dry_reference"
RESULTS_DIR = DATA_ROOT / "results"

DEFAULT_FABRICS = [
    "cotton",
    "denim",
    "shirt_cotton",
    "tshirt_cotton",
    "silk",
    "nylon",
    "two_layer",
    "three_layer",
]


def sanitize_name(name: str) -> str:
    """Normalize fabric or session name to safe lowercase snake_case."""
    cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[-\s]+", "_", cleaned)


def get_all_fabrics() -> list[str]:
    """Retrieve all fabric names discovered in raw/ or dry_reference/ or default list."""
    fabrics = set(DEFAULT_FABRICS)
    if RAW_DIR.exists():
        for p in RAW_DIR.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                fabrics.add(p.name)
    if DRY_REF_DIR.exists():
        for p in DRY_REF_DIR.iterdir():
            if p.is_file() and p.name.endswith(("_dry.jpg", "_dry.jpeg", "_dry.png")):
                fab_name = re.sub(r"_dry\.(jpg|jpeg|png)$", "", p.name)
                if fab_name:
                    fabrics.add(fab_name)
    return sorted(fabrics)


def find_dry_reference(fabric: str) -> Optional[Path]:
    """Find the dry reference file for a given fabric."""
    if not DRY_REF_DIR.exists():
        return None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = DRY_REF_DIR / f"{fabric}_dry{ext}"
        if candidate.exists() and candidate.stat().st_size > 10:
            return candidate
    return None


def get_test_images_for_fabric(fabric: str) -> list[Path]:
    """Get all test images for a fabric, including those in test run subfolders."""
    fab_dir = RAW_DIR / fabric
    if not fab_dir.exists():
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images: list[Path] = []
    for p in fab_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts and p.stat().st_size > 10:
            images.append(p)
    return sorted(images)


def draw_detections_on_image(
    image_bgr: np.ndarray, detections: list
) -> np.ndarray:
    """Draw circles, center dots, and confidence labels on an image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    line_thick = max(1, int(round(min(h, w) / 500)))
    font_scale = max(0.4, min(h, w) / 1200)

    for det in detections:
        cx = int(round(det.x_px))
        cy = int(round(det.y_px))
        r = int(round(det.radius_px))

        # Green droplet circle
        cv2.circle(annotated, (cx, cy), r, (0, 230, 0), line_thick + 1)
        # Center red dot
        cv2.circle(annotated, (cx, cy), max(2, line_thick + 1), (0, 0, 255), -1)

        # Label box with droplet number and confidence
        label = f"#{det.droplet_number}: {det.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thick
        )
        lx = max(0, cx - label_w // 2)
        ly = max(label_h + 5, cy - r - 5)

        cv2.rectangle(
            annotated,
            (lx - 2, ly - label_h - 2),
            (lx + label_w + 2, ly + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 128),
            line_thick,
            cv2.LINE_AA,
        )

    return annotated


def main() -> None:
    st.set_page_config(
        page_title="Droplet Detector Dashboard",
        page_icon="💧",
        layout="wide",
    )

    st.title("💧 Droplet Detector Dashboard")
    st.caption(
        "Classical computer-vision pipeline for automated water-droplet detection on fabric specimens"
    )

    # Sidebar settings
    st.sidebar.header("⚙️ Detection Configuration")
    sensitivity = st.sidebar.slider(
        "Sensitivity Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence threshold. Lower values detect more droplets but may include more false positives.",
    )
    diff_noise_floor = st.sidebar.slider(
        "Diff Noise Floor",
        min_value=5,
        max_value=50,
        value=15,
        step=1,
        help="Differences below this intensity are suppressed as sensor noise.",
    )
    min_diameter_mm = st.sidebar.number_input(
        "Min Droplet Diameter (mm)",
        min_value=0.2,
        max_value=10.0,
        value=1.0,
        step=0.2,
    )
    max_diameter_mm = st.sidebar.number_input(
        "Max Droplet Diameter (mm)",
        min_value=0.5,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Overview")
    fabrics_list = get_all_fabrics()
    st.sidebar.write(f"Total Fabrics Registered: **{len(fabrics_list)}**")

    # Tabs
    tab_ref, tab_raw, tab_detect, tab_data = st.tabs([
        "📷 1. Upload Dry Reference",
        "🧪 2. Upload Wet/Test Photos",
        "🔍 3. Run Detection & Results",
        "📁 4. Dataset Manager",
    ])

    # -------------------------------------------------------------
    # TAB 1: Upload Dry Reference
    # -------------------------------------------------------------
    with tab_ref:
        st.subheader("Upload Dry Reference Photo")
        st.info(
            "Upload a photo of the dry fabric taken before water is applied. "
            "This dry reference cancels weave patterns and fixed reflections during differencing."
        )

        col_select, col_new = st.columns([1, 1])
        with col_select:
            fabric_options = ["-- Select Fabric --"] + fabrics_list + ["➕ Add New Fabric..."]
            selected_choice = st.selectbox("Choose Fabric Type", fabric_options, key="ref_fabric_select")

        new_fabric_input = ""
        if selected_choice == "➕ Add New Fabric...":
            with col_new:
                new_fabric_input = st.text_input(
                    "Enter New Fabric Name",
                    placeholder="e.g. corduroy, microfiber",
                    key="ref_new_fabric_input",
                )

        target_fabric = ""
        if selected_choice == "➕ Add New Fabric...":
            target_fabric = sanitize_name(new_fabric_input)
        elif selected_choice != "-- Select Fabric --":
            target_fabric = selected_choice

        if target_fabric:
            st.write(f"Selected Fabric: **`{target_fabric}`**")

            # Check if one already exists
            existing_ref = find_dry_reference(target_fabric)
            if existing_ref:
                st.success(f"Existing dry reference found: `{existing_ref.name}`")
                with st.expander("View existing dry reference"):
                    st.image(Image.open(existing_ref), caption=f"Current dry reference: {target_fabric}")

            dry_file = st.file_uploader(
                f"Upload Dry Reference Image for '{target_fabric}'",
                type=["jpg", "jpeg", "png"],
                key="dry_file_uploader",
            )

            if dry_file is not None:
                st.image(dry_file, caption=f"Preview: {dry_file.name}", width=400)
                if st.button("Save Dry Reference Photo", type="primary", key="save_dry_ref_btn"):
                    DRY_REF_DIR.mkdir(parents=True, exist_ok=True)
                    ext = Path(dry_file.name).suffix.lower()
                    if ext not in (".jpg", ".jpeg", ".png"):
                        ext = ".jpg"
                    dest_file = DRY_REF_DIR / f"{target_fabric}_dry{ext}"

                    dest_file.write_bytes(dry_file.getbuffer())
                    st.success(f"✅ Successfully saved dry reference to: `{dest_file}`")
                    st.rerun()

    # -------------------------------------------------------------
    # TAB 2: Upload Raw Test Photos (with droplets)
    # -------------------------------------------------------------
    with tab_raw:
        st.subheader("Upload Raw Test Photos (with droplets)")
        st.markdown(
            "> **Why group by fabric?** The detector pairs each test photo with the corresponding "
            "dry reference photo of that fabric. Organizing photos by fabric and test run guarantees "
            "accurate differencing and organized reporting."
        )

        col_fab, col_batch = st.columns([1, 1])
        with col_fab:
            raw_fabric_choice = st.selectbox(
                "Fabric Type for these test photos",
                ["-- Select Fabric --"] + fabrics_list + ["➕ Add New Fabric..."],
                key="raw_fabric_select",
            )
        with col_batch:
            test_batch = st.text_input(
                "Optional: Test Batch / Session Name",
                value="",
                placeholder="e.g. test_001, run_1 (leave empty for fabric root)",
                help="If specified, images will be saved inside data/raw/<fabric>/<batch_name>/",
                key="raw_batch_input",
            )

        raw_fabric = ""
        if raw_fabric_choice == "➕ Add New Fabric...":
            new_fab = st.text_input("Enter New Fabric Name", key="raw_new_fab_input")
            raw_fabric = sanitize_name(new_fab)
        elif raw_fabric_choice != "-- Select Fabric --":
            raw_fabric = raw_fabric_choice

        if raw_fabric:
            clean_batch = sanitize_name(test_batch) if test_batch.strip() else ""
            target_folder = RAW_DIR / raw_fabric
            if clean_batch:
                target_folder = target_folder / clean_batch

            st.write(f"Target Save Directory: **`{target_folder}`**")

            test_files = st.file_uploader(
                f"Select photo(s) with droplets for '{raw_fabric}'",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True,
                key="raw_files_uploader",
            )

            if test_files:
                st.write(f"**{len(test_files)}** file(s) selected.")
                if st.button("Save All Uploaded Photos", type="primary", key="save_raw_btn"):
                    target_folder.mkdir(parents=True, exist_ok=True)
                    saved_count = 0
                    for f in test_files:
                        dest = target_folder / f.name
                        dest.write_bytes(f.getbuffer())
                        saved_count += 1
                    st.success(f"✅ Successfully saved {saved_count} photo(s) to `{target_folder}`")
                    st.rerun()

    # -------------------------------------------------------------
    # TAB 3: Run Detection & Live Inspection
    # -------------------------------------------------------------
    with tab_detect:
        st.subheader("Run Detection & Visual Inspection")

        detect_fabric = st.selectbox(
            "Select Fabric to Test",
            ["-- Select Fabric --"] + fabrics_list,
            key="detect_fabric_select",
        )

        if detect_fabric and detect_fabric != "-- Select Fabric --":
            dry_ref_path = find_dry_reference(detect_fabric)
            test_images = get_test_images_for_fabric(detect_fabric)

            col_status1, col_status2 = st.columns([1, 1])
            with col_status1:
                if dry_ref_path:
                    st.success(f"✅ Dry Reference: `{dry_ref_path.name}`")
                else:
                    st.error(
                        f"❌ No dry reference found for `{detect_fabric}`! "
                        "Please upload one in Tab 1 before running detection."
                    )
            with col_status2:
                st.info(f"📁 Available Test Photos: **{len(test_images)}** found")

            if dry_ref_path and test_images:
                test_image_choice = st.selectbox(
                    "Choose Test Image",
                    test_images,
                    format_func=lambda p: str(p.relative_to(RAW_DIR / detect_fabric)),
                    key="detect_test_image_select",
                )

                col_preview_dry, col_preview_test = st.columns(2)
                with col_preview_dry:
                    st.markdown("**Dry Reference**")
                    st.image(Image.open(dry_ref_path), use_container_width=True)
                with col_preview_test:
                    st.markdown(f"**Test Image: {test_image_choice.name}**")
                    st.image(Image.open(test_image_choice), use_container_width=True)

                if st.button("🚀 Run Droplet Detection", type="primary", key="run_detect_btn"):
                    with st.spinner("Processing image through CV pipeline..."):
                        try:
                            config = DropletDetectorConfig(
                                results_save_path=RESULTS_DIR / "reports",
                                image_save_path=RESULTS_DIR / "images",
                                sensitivity=sensitivity,
                                diff_noise_floor=diff_noise_floor,
                                droplet_min_diameter_mm=min_diameter_mm,
                                droplet_max_diameter_mm=max_diameter_mm,
                            )

                            current_bgr = cv2.imread(str(test_image_choice))
                            dry_bgr = cv2.imread(str(dry_ref_path))

                            # Ensure dry reference matches current dimensions
                            if dry_bgr.shape[:2] != current_bgr.shape[:2]:
                                if (dry_bgr.shape[0], dry_bgr.shape[1]) == (current_bgr.shape[1], current_bgr.shape[0]):
                                    dry_bgr = cv2.rotate(dry_bgr, cv2.ROTATE_90_CLOCKWISE)
                                if dry_bgr.shape[:2] != current_bgr.shape[:2]:
                                    dry_bgr = cv2.resize(
                                        dry_bgr, (current_bgr.shape[1], current_bgr.shape[0]), interpolation=cv2.INTER_AREA
                                    )

                            # Run pipeline
                            detections = detect_droplets_in_image(
                                str(test_image_choice),
                                str(dry_ref_path),
                                config,
                            )

                            # Create diff image for visualization
                            diff_img = diff_against_dry_reference(
                                current_bgr, dry_bgr, noise_floor=diff_noise_floor
                            )

                            # Annotated result
                            annotated_bgr = draw_detections_on_image(current_bgr, detections)
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                            st.markdown("---")
                            st.subheader("🎯 Detection Results")

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Droplets Detected", len(detections))
                            m2.metric("Sensitivity", f"{sensitivity:.2f}")
                            m3.metric("Noise Floor", diff_noise_floor)

                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.markdown("**Annotated Test Image (Detected Droplets)**")
                                st.image(annotated_rgb, use_container_width=True)
                            with res_col2:
                                st.markdown("**Noise-Suppressed Difference Map**")
                                st.image(diff_img, use_container_width=True, clamp=True)

                            if detections:
                                st.subheader("📋 Detected Droplets Table")
                                rows = []
                                for d in detections:
                                    rows.append({
                                        "Droplet #": d.droplet_number,
                                        "X (px)": round(d.x_px, 1),
                                        "Y (px)": round(d.y_px, 1),
                                        "Radius (px)": round(d.radius_px, 1),
                                        "Confidence": f"{d.confidence * 100:.1f}%",
                                    })
                                st.dataframe(rows, use_container_width=True)

                                # Save result option
                                json_data = json.dumps(
                                    [d.model_dump(mode="json") for d in detections],
                                    indent=2,
                                )
                                st.download_button(
                                    label="📥 Download Detections (JSON)",
                                    data=json_data,
                                    file_name=f"{test_image_choice.stem}_detections.json",
                                    mime="application/json",
                                )
                            else:
                                st.warning("No droplets detected above the current sensitivity threshold.")

                        except Exception as e:
                            st.error(f"Detection failed with error: {e}")

            elif not test_images and dry_ref_path:
                st.warning(f"No test photos uploaded yet for `{detect_fabric}`. Go to Tab 2 to upload test photos.")

    # -------------------------------------------------------------
    # TAB 4: Dataset Manager
    # -------------------------------------------------------------
    with tab_data:
        st.subheader("📁 Dataset & Library Status")
        st.write("Summary of all registered fabrics, reference status, and test image counts:")

        table_data = []
        for fab in fabrics_list:
            ref = find_dry_reference(fab)
            imgs = get_test_images_for_fabric(fab)
            table_data.append({
                "Fabric": fab,
                "Dry Reference": f"✅ {ref.name}" if ref else "❌ Missing",
                "Test Photos Count": len(imgs),
                "Directory": f"data/raw/{fab}",
            })

        st.dataframe(table_data, use_container_width=True)


if __name__ == "__main__":
    main()
