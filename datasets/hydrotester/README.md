# Hydrotester training examples

These archives contain the exact image and YOLO label pairs used in the two
project training runs behind the bundled
[`Visible_Leaks_Trained.pt`](../../models/fabrics/Visible_Leaks_Trained.pt)
checkpoint. The YOLO class is `0` (`water_droplet`).

| Archive | Train | Validation | Use |
| --- | ---: | ---: | --- |
| [Initial detector](initial-training.zip) | 65 frames | 13 frames | Prepare the initial detector from the repository's `yolo11n.pt`. The labels were provisionally reviewed and should receive expert review before acceptance testing. |
| [Circular fine-tuning](circular-fine-tuning.zip) | 6 frames | 2 frames | Fine-tune the initial detector using the same masked fabric view as replay. One additional saved frame was skipped when the rim could not be located. |

Both archives contain `images/train`, `images/val`, `labels/train`, and
`labels/val`. Empty label files represent background examples. Import a ZIP
under **Train the Model → Import labelled CVAT ZIP** to use its existing split.
The dashboard starts training from the active checkpoint.

The verified fine-tuning run used eight passes at 640 pixels with batch size 4,
AdamW at 0.0002, and ten frozen layers. It matched both held-back boxes and
confirmed two marked positions at 98.51 s and 146.71 s in the selected
recording. The [ten demo videos](../../demo-videos/) are for demonstration.
Use independently reviewed recordings to measure missed detections, false alarms,
and timing errors on unfamiliar specimens.
