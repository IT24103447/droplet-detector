# Current model test on Visible leaks

**45 original recordings tested with `models/droplet-seg.pt` at the dashboard's 0.25 confidence threshold.**

- **45/45** filled all three leak slots in the opening eight frames using the full frame and the default tracker. In **41/45**, all three were outside the estimated specimen area.
- **12/45** had at least one clear droplet match in the reviewed fabric-crop frames.
- **4/45** had matches on pooled water, with no separate droplet match verified.
- **11/45** produced boxes in the fabric area without a verified match. **18/45** had no candidate centred in the fabric area in any sampled frame.

On `100826-1157`, all three opening hits were outside the specimen. The right-side droplet is visible by about 5 s. It remained unmarked in the cropped pass. The model instead drew boxes on the seam. See the [1157 frame sheet](080.jpg).

## How the test was run

The opening check used the same weights, confidence threshold, three-frame confirmation rule and ten-pixel tracking distance as the dashboard. For the second check, the model ran on a specimen-centred crop across each recording. An approximate ellipse classified candidate centres as inside or outside the fabric in both checks. Its edge is approximate because the specimen and fixture vary by clip.

The crop check used **1,538 sampled frames**. I visually inspected the prediction sheets for clear overlaps with droplets or pooled water. Yellow boxes have centres within the estimated fabric area. Red boxes are outside it. The times below are examples of where I verified a match. This was a positive-only screening test with sampled video frames, so it does not measure false-positive or missed-detection rates, verify the first three leak positions, or establish detection timing accuracy.

No originals were moved or copied. The model and application code were unchanged.

## Results by video

Select a filename to see the sampled frames and model boxes.

| Video | Opening slots outside fabric | Crop frames with a candidate | Visual review | Example match |
|---|---:|---:|---|---:|
| [Custom__10_0_PSI_min-2026-10-19_10_13-video-101926-1014.mp4](011.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__10_0_PSI_min-2026-10-19_12_07-video-101926-1207.mp4](020.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__10_0_PSI_min-2026-10-20_11_09-video-102026-1111.mp4](032.jpg) | 3/3 | 1/40 | Visible droplet matched | 114.4 s |
| [Custom__10_0_PSI_min-2026-10-20_11_15-video-102026-1117.mp4](034.jpg) | 3/3 | 16/30 | Visible droplet matched | 37.9 s |
| [Custom__10_0_PSI_min-2026-10-20_11_36-video-102026-1137.mp4](040.jpg) | 3/3 | 0/28 | No candidate centre in fabric area | — |
| [Custom__10_0_PSI_min-pinnacal_15mm_hood-video-102026-1221.mp4](061.jpg) | 3/3 | 1/28 | Candidate boxes, no verified match | — |
| [Custom__5_00_PSI_min-2026-10-07_06_11-video-100726-0631.mp4](077.jpg) | 3/3 | 26/53 | Pooled water matched | 981.1 s |
| [Custom__5_00_PSI_min-2026-10-08_11_53-video-100826-1157.mp4](080.jpg) | 3/3 | 38/62 | Candidate boxes, no verified match | — |
| [Custom__5_00_PSI_min-2026-10-09_04_42-video-100926-0444.mp4](084.jpg) | 2/3 | 23/34 | Candidate boxes, no verified match | — |
| [Custom__5_00_PSI_min-2026-10-09_05_03-video-100926-0505.mp4](085.jpg) | 3/3 | 3/39 | Candidate boxes, no verified match | — |
| [Custom__5_00_PSI_min-2026-10-09_05_06-video-100926-0511.mp4](086.jpg) | 3/3 | 12/53 | Pooled water matched | 240.0 s |
| [Custom__5_00_PSI_min-2026-10-09_07_15-video-100926-0716.mp4](094.jpg) | 3/3 | 2/22 | Visible droplet matched | 17.4 s |
| [Custom__5_00_PSI_min-2026-10-09_09_23-video-100926-0923.mp4](100.jpg) | 3/3 | 19/22 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-13_07_02-video-101326-0704.mp4](120.jpg) | 3/3 | 1/33 | Visible droplet matched | 73.5 s |
| [Custom__5_0_PSI_min-2026-10-14_04_16-video-101426-0420.mp4](127.jpg) | 2/3 | 29/29 | Visible droplet matched | 68.2 s |
| [Custom__5_0_PSI_min-2026-10-14_04_21-video-101426-0423.mp4](128.jpg) | 3/3 | 21/22 | Visible droplet matched | 31.0 s |
| [Custom__5_0_PSI_min-2026-10-14_05_20-video-101426-0525.mp4](132.jpg) | 3/3 | 0/53 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-14_06_16-video-101426-0617.mp4](136.jpg) | 3/3 | 0/21 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-14_06_38-video-101426-0639.mp4](140.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-14_09_04-video-101426-0918.mp4](145.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-14_09_29-video-101426-0932.mp4](146.jpg) | 3/3 | 0/43 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-14_09_32-video-101426-0934.mp4](147.jpg) | 3/3 | 0/30 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-14_10_22-video-101426-1026.mp4](149.jpg) | 3/3 | 15/45 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-14_10_28-video-101426-1029.mp4](150.jpg) | 3/3 | 16/29 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-15_05_41-video-101526-0546.mp4](155.jpg) | 3/3 | 6/53 | Visible droplet matched | 212.4 s |
| [Custom__5_0_PSI_min-2026-10-15_05_49-video-101526-0558.mp4](156.jpg) | 3/3 | 4/45 | Visible droplet matched | 112.7 s |
| [Custom__5_0_PSI_min-2026-10-15_05_59-video-101526-0601.mp4](157.jpg) | 3/3 | 1/30 | Visible droplet matched | 65.3 s |
| [Custom__5_0_PSI_min-2026-10-15_06_37-video-101526-0638.mp4](159.jpg) | 3/3 | 20/22 | Pooled water matched | 36.6 s |
| [Custom__5_0_PSI_min-2026-10-15_12_37-video-101526-1239.mp4](166.jpg) | 3/3 | 0/44 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-15_12_45-video-101526-1248.mp4](167.jpg) | 3/3 | 22/48 | Visible droplet matched | 152.7 s |
| [Custom__5_0_PSI_min-2026-10-16_10_14-video-101626-1024.mp4](174.jpg) | 2/3 | 18/22 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-16_15_01-video-101626-1511.mp4](184.jpg) | 3/3 | 23/23 | Visible droplet matched | 43.1 s |
| [Custom__5_0_PSI_min-2026-10-17_09_24-video-101726-0929.mp4](189.jpg) | 3/3 | 2/52 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-17_09_53-video-101726-1015.mp4](191.jpg) | 3/3 | 0/29 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-17_11_10-video-101726-1115.mp4](195.jpg) | 3/3 | 14/37 | Visible droplet matched | 106.1 s |
| [Custom__5_0_PSI_min-2026-10-17_13_15-video-101726-1319.mp4](198.jpg) | 3/3 | 4/53 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-19_09_17-video-101926-0919.mp4](202.jpg) | 3/3 | 38/40 | Candidate boxes, no verified match | — |
| [Custom__5_0_PSI_min-2026-10-19_09_36-video-101926-0940.mp4](204.jpg) | 2/3 | 6/53 | Pooled water matched | 233.3 s |
| [Custom__5_0_PSI_min-2026-10-19_09_47-video-101926-0949.mp4](205.jpg) | 3/3 | 0/39 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-19_13_37-video-101926-1337.mp4](211.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-19_14_28-video-101926-1429.mp4](226.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-19_15_54-video-101926-1556.mp4](239.jpg) | 3/3 | 0/33 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-19_16_49-video-101926-1649.mp4](241.jpg) | 3/3 | 0/20 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-20_08_29-video-102026-0830.mp4](255.jpg) | 3/3 | 0/22 | No candidate centre in fabric area | — |
| [Custom__5_0_PSI_min-2026-10-20_08_31-video-102026-0832.mp4](256.jpg) | 3/3 | 0/25 | No candidate centre in fabric area | — |
