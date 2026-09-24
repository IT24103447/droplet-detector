# One-Image Water-Droplet Model Guide

This guide explains how to prepare the local water-droplet model. The final
system uses **one fabric photo at a time**. It does not need a dry reference
photo when detecting droplets.

## What the application does

The dashboard has three tabs, used in this order:

1. **Label training photos** — teach the model what a real water droplet looks
   like.
2. **Train the model** — build one local model from those labelled examples.
3. **Detect one photo** — upload a new fabric photo and receive the droplet
   count and marked image.

## Step 1: Start the dashboard

From the project folder, run:

```bash
python scripts/run_ui.py
```

Open the local web page shown by the command.

## Step 2: Label training photos

You can use the built-in dashboard labelling screen, but **CVAT is the
recommended option** when you have many images or the in-app rectangle tool is
not working reliably. Both options produce the same kind of training labels.

### Recommended: label images in CVAT

CVAT is a separate image-labelling tool. It stores rectangles as annotation
data; it does **not** permanently draw them onto the photo.

1. Create a CVAT project called `Droplet Detector`.
2. Add one label only: `water_droplet`.
3. Split your original, clean photos before uploading:
   - about 80% for **Training**;
   - about 20% for **Validation**.
4. Create a CVAT task for the training photos and upload the clean originals.
   Do **not** upload images with circles or rectangles already drawn on them.
5. Select the `water_droplet` label and draw one tight **rectangle** around
   every genuine droplet in each image.
6. Leave dry/no-water photos with no rectangles. These are valid negative
   examples and help the model ignore normal cloth texture.
7. Export the completed task as **Ultralytics YOLO Detection 1.0**.
8. Repeat the same process in a separate CVAT task for validation photos.

Extract the CVAT exports so that images and matching annotation files have
this structure:

```text
datasets/
  images/
    train/                 # original training photos
    val/                   # original validation photos
  labels/
    train/                 # matching CVAT .txt labels
    val/                   # matching CVAT .txt labels
```

For example, each training image must have a matching text file:

```text
datasets/images/train/photo_01.jpg
datasets/labels/train/photo_01.txt
```

Do not use the `test` folder yet. The current training command needs the
Training and Validation folders only.

### Alternative: use the dashboard labelling screen

Open **1. Label training photos**.

1. Choose either **Training set** or **Validation set**.
2. Upload one photo of a fabric.
3. If the photo contains water droplets, draw one rectangle tightly around
   each real droplet.
4. If the photo contains no water droplets, draw no rectangles.
5. Click **Save labelled example**.

The application saves the image and labels into the correct project folders
automatically. Do not manually move files into dataset folders.

### What should be labelled?

Label only genuine water droplets.

Do not label:

- cloth weave or texture;
- reflections that are not water;
- folds, seams, shadows, or lint;
- existing stains that are not water droplets.

Photos with no droplets are important. They teach the model that a normal
fabric texture is not a leak.

### Do I need to choose a fabric type?

No. Upload photos without grouping them by cotton, denim, or another fabric
name. The model learns from the visual appearance of the image.

It is still useful to keep a separate personal note of which fabric was used
in each test run, so the team can later measure accuracy across fabric types.

## Step 3: Training set versus validation set

The **training set** teaches the model. The **validation set** checks whether
the trained model works on photos it has not seen before.

Use this simple rule:

- Put about 80% of labelled photos in **Training set**.
- Put about 20% of labelled photos in **Validation set**.

Do not put near-identical frames from the same moment in both sets. For
example, put photos from one recording in Training and photos from a different
recording in Validation.

## Step 4: What photos should the team collect?

Include examples from at least three representative fabric types and include:

- dry fabric with no water;
- one, two, and several water droplets;
- small and large droplets;
- reflections and shiny regions;
- folds, seams, shadows, and machine vibration;
- different lighting conditions;
- frames where an existing droplet grows.

The model can only learn examples that the team provides. More varied,
correctly labelled examples make it more reliable on new clothes and fabrics.

## Step 5: Train the local model

After there are labelled photos in both Training and Validation sets, open
**2. Train the model**. Run these commands in a terminal from the project
folder:

```bash
pip install -e ".[ml]"
python scripts/train_detection.py
```

Training creates a file named `best.pt`, usually under a `runs/detect/...`
folder. Copy that file to:

```text
models/droplet-seg.pt
```

## Step 6: Detect droplets in a new photo

Open **3. Detect one photo**.

1. Upload one new fabric image.
2. Click **Detect water droplets**.
3. The application shows the number of detected droplets and marks each one
   on the image.

No dry reference image is required in this step.

## Important reminder

The one-image model is not ready until it has been trained. Before training,
the detection tab correctly shows a message saying that no trained model is
available. This is expected.
