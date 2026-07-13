# LiveFISH Nucleus Cropping — Usage & Debugging Guide

This covers how to run the two scripts that make up Step 1 of the LiveFISH pipeline:
`crop_nuclei_sam.py` segments nuclei and writes a JSON sidecar per FOV file;
`save_crops.py` reads those JSONs and writes the actual TIFF crops.
Keeping the two steps separate means you can re-save TIFFs with different settings
(e.g. different LUT mode) without re-running the slow µSAM segmentation.

The canonical Windows tutorial, locked parameter table, GPU setup, stable cell
IDs, and troubleshooting are maintained in [`../README.md`](../README.md) and
[`../REQUIREMENTS_ND2_TO_TRAJ.md`](../REQUIREMENTS_ND2_TO_TRAJ.md). The commands
below are a compact cross-platform reference and use the same production files.

---

## Setup

### Prerequisites

- Python 3.13 (validated version)
- The pinned packages in `requirements_nd2_to_traj.txt`
- Optional NVIDIA CUDA or Apple MPS acceleration; CPU is supported

### Install dependencies

Create an isolated venv from the repository root:

```bash
python3.13 -m venv .venv_nd2_to_traj
source .venv_nd2_to_traj/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_nd2_to_traj.txt
```

For NVIDIA setup, use the official PyTorch selector procedure documented in
`REQUIREMENTS_ND2_TO_TRAJ.md`; never edit the Python source to select a device.
The CLI accepts `--device auto`, `cuda`, `mps`, or `cpu`.

### Navigate to the project folder

All paths below are relative to this repository root.
Open a terminal and `cd` there first:

```bash
cd "/path/to/OligoLiveFish-ML-ZZH" # REPLACE: local repository path
```

### Note on Google Drive paths

If your data lives in a Google Drive for Desktop shortcut, mark the folder
**"Available offline"** in Finder before running. Otherwise the first read of
each file will be bottlenecked by Drive's on-demand download speed, and a
mid-run sleep or sync interruption can stall the script.

---

## Step 1 — Segment nuclei (`crop_nuclei_sam.py`)

### Single file (recommended starting point)

Always start with one file and check the visualizations before running the full batch.
Use a small or known-good file first.

```bash
ND2="/path/to/<ND2_stem>.nd2"       # REPLACE: one real ND2 file
CROP_ROOT="/path/to/cell_crops"     # REPLACE: output parent
python nucleus_segmentation/crop_nuclei_sam.py "$ND2" \
    --device auto \
    --nucleus-channel 0 --margin 30 \
    --min-area 1000 --max-area 200000 \
    --segmentation-mode apg --model-type vit_b_lm \
    --border-margin 5 --mask-border-margin 0 \
    --output-root "$CROP_ROOT"
```

### Full batch (all .nd2 files in folder)

Pass the folder instead of a single file — the script finds all `.nd2` files recursively.
Point at the root data folder to process everything at once:

```bash
RAW_DIR="/path/to/raw_nd2"            # REPLACE: recursively searched input folder
python nucleus_segmentation/crop_nuclei_sam.py "$RAW_DIR" \
    --device cuda \
    --nucleus-channel 0 --margin 30 \
    --min-area 1000 --max-area 200000 \
    --segmentation-mode apg --model-type vit_b_lm \
    --border-margin 5 --mask-border-margin 0 \
    --output-root "$CROP_ROOT"
```

### Nested folder structures

Pointing the script at a folder will recursively find all `.nd2` files in subfolders.
With `--output-root`, one output folder named after each ND2 stem is written
under that output parent. Without it, output is written beside each ND2.

### Processing while data is still downloading

If files are still syncing from Google Drive (or being copied in via rsync), you can
process individual completed files in parallel with the transfer. Use:

```bash
find "<data dir>" -name "*.nd2" -mmin +5
```

to list `.nd2` files that haven't been modified in the last 5 minutes — those are
very likely fully-transferred. Process those individually rather than pointing the
script at the parent folder, since a folder-mode run could try to read a file
mid-transfer and crash on it (or worse, succeed on a truncated file). Once the
transfer finishes, do the full-folder batch run for everything still unprocessed.

### Parameters

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--nucleus-channel` | `0` | Which channel to segment on (0 = DAPI/nucleus stain) |
| `--margin` | `30` | Padding in pixels added around each nucleus bounding box |
| `--min-area` | `1000` | Minimum nucleus area in pixels — smaller masks discarded as debris |
| `--max-area` | `200000` | Maximum nucleus area in pixels — larger masks discarded |
| `--segmentation-mode` | `apg` | APG (recommended for touching nuclei); AMG is the plain SAM alternative |
| `--model-type` | `vit_b_lm` | µSAM model fine-tuned on fluorescence microscopy; `vit_l_lm` is larger and slower for marginal gain on clean nuclei |
| `--border-margin` | `5` | Min distance (px) from image border to nucleus centroid; smaller masks at edges discarded |
| `--mask-border-margin` | `-1` | Reject a mask whose pixels touch this edge margin; locked workflow uses `0` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu`; use explicit CUDA after a successful GPU preflight for a large batch |
| `--output-root` | ND2 parent | Parent containing one output folder per ND2 stem |

### Outputs (per .nd2 file)

Results are written next to each input file, in a folder named after the file stem:

```
data for analysis/FOV (.nd2 files)/<stem>/
├── <stem>_crops.json              ← bbox + suppression coords consumed by save_crops.py
├── <stem>_mask_1.tif              ← binary nuclear mask: (T, cropH, cropW) uint8, cropped to the nucleus bbox, nucleus=255 background=0, same mask repeated across all T frames
├── <stem>_mask_2.tif
├── ...                            ← one mask TIFF per accepted nucleus, numbered to match crop TIFFs
└── visualizations/
    ├── seg_overview.png           ← 4-panel: raw image | µSAM raw | after filter+merge | final
    ├── crop_grid.png              ← thumbnail of every accepted nucleus crop
    ├── suppression_demo.png       ← before/after neighbour suppression for first 6 crops
    └── all_channels_demo.png      ← all 4 channels for first 4 crops
```

**Always open `seg_overview.png` first** — it's the fastest way to tell if segmentation worked.

---

## Step 2 — Save TIFFs (`save_crops.py`)

Recursively finds all `*_crops.json` files under the given directory and writes one
TIFF + `_metadata.json` sidecar per crop. Run this after Step 1 completes.

```bash
FOV_CROP_DIR="$CROP_ROOT/<ND2_stem>" # REPLACE <ND2_stem>: Step 1 output folder
python nucleus_segmentation/save_crops.py "$FOV_CROP_DIR"
```

### Outputs (per .nd2 file)

```
<stem>/
├── <stem>_1.tif                ← (T, Z, C, Y, X) uint16, neighbours zeroed, per-channel LUTs embedded
├── <stem>_1_metadata.json      ← acquisition metadata not storable in ImageJ TIFF format
├── <stem>_2.tif
├── <stem>_2_metadata.json
└── ...
```

TIFFs open in Fiji with per-channel colors matching the original .nd2. Each file is a
(T, Z, C, Y, X) hyperstack ready for the downstream
`trajectory_extraction/run_full_pipeline_v3.py` trajectory-extraction runner.

Each `_metadata.json` sidecar contains:
- `source_nd2` — absolute path back to the originating FOV file
- `acquisition_date` — date/time string from NIS-Elements
- `crop_shape` — `{T, Z, C, Y, X}` of the saved TIFF
- `bbox` — `{r0, r1, c0, c1}` pixel coordinates of the crop within the original FOV
- `pixel_size` — XY and Z voxel size in µm
- `time` — frame interval in seconds, fps, and number of frames
- `acquisition` — objective name, NA, magnification, modality
- `channels` — per-channel: name, emission wavelength (nm), exposure time (ms), display min/max

Run Step 2 after Step 1 — it can also be re-run alone to regenerate TIFFs from existing JSONs
(e.g. if TIFF export settings change) without re-running the slow µSAM segmentation.

---

## Debugging

The visualizations are your main diagnostic tool. Here's how to read the common failure modes:

### "Too many crops — there's obvious debris or tiny fragments in the grid"

Open `seg_overview.png` and look at panel 3 (after filter+merge). If small bright spots
or partial cells at the image edge are making it through, raise `--min-area`.
Try 2000 first, then 3000 if debris persists. The right value is where the crop grid
shows only clean oval nuclei.

### "Too few crops — some nuclei are clearly missing"

If dim or small nuclei are being dropped, lower `--min-area` (try 500).
If nuclei near the image border are missing, that's intentional — the border filter drops any nucleus whose centroid is within `--border-margin` px of the edge (default 5) to avoid partial crops. Raise this if you want to be more conservative about edge nuclei, lower to 0 to keep them all.

### "Two touching nuclei are showing up as one crop"

APG mode handles most touching pairs automatically via watershed splitting.
If a merged pair is still slipping through, look at the solidity of the combined mask —
if it's above `MIN_SOLIDITY` (0.70 in the script), the split won't even be attempted.
Lower `MIN_SOLIDITY` slightly (try 0.65) to trigger the split on more masks.
If the split is being attempted but failing, lower `SPLIT_MIN_DIST` (currently 20 px)
so the peak detector can find two centres that are closer together.

### "One nucleus is split into two separate crops"

This means µSAM fragmented a single nucleus into two masks. `merge_adjacent_masks`
should catch this, but it only merges masks within `MERGE_PROXIMITY` px of each other
(currently 2 px). If the gap between fragments is larger, raise `MERGE_PROXIMITY` (try 5–10).
Check `seg_overview.png` panel 2 (µSAM raw) to confirm it's fragmentation and not
two genuinely separate nuclei that happen to be close.

### "The target nucleus is partially zeroed in a crop"

The neighbour suppression is zeroing pixels that belong to the target nucleus.
This usually means µSAM drew the mask boundary too tightly, leaving some nucleus pixels
outside the mask — those pixels then get zeroed as "not belonging to any nucleus."
Increasing `--margin` won't fix this; the issue is the mask boundary itself.
Check `suppression_demo.png` to confirm. If it's systematic, try `--model-type vit_l_lm`
(larger model, better boundaries) or lower `MIN_SOLIDITY` in the script to let the
watershed refine the boundary.

### "A neighbouring nucleus is visible in a crop (not zeroed)"

This means the neighbour's mask wasn't accepted by the filter — it was dropped as too
small, too large, or a border nucleus — so there was nothing to suppress it with.
Check `seg_overview.png` panel 3 to see if the offending neighbour has a mask at all.
If not, adjust `--min-area` or `--border-margin` so it gets accepted.

### "Script crashes on one file but works on others"

Each file is wrapped in a try/except so the batch continues — check stderr for the
traceback. Common causes: corrupted .nd2 file, unexpected axis order in the nd2 metadata,
a file with a different number of channels than expected, or a file that was still
being transferred when the script started reading it. The axes string is printed
on load (`nd2 axes string: ...`) — verify it matches `TZCYX` or a close variant.

---

## Repeatable results checklist

1. Use the same named conda environment every time — packages must match exactly.
2. Run single-file first, inspect `seg_overview.png` and `crop_grid.png` before batching.
3. **Record the parameters you used** — the `_crops.json` internal sidecars (Step 1 output, consumed by Step 2) do not store segmentation parameters. The `_metadata.json` sidecars (Step 2 output, for downstream use) store acquisition metadata but also do not store segmentation parameters.
4. If you change parameters, re-run Step 1 before Step 2. `save_crops.py` reads from
   the JSON sidecars, so TIFFs reflect whatever segmentation last wrote them. If a
   Step 1 run crashed partway through, JSONs from files that didn't get re-processed
   will still hold old parameters — delete those specific JSONs or re-run those files
   before running Step 2, or you'll get TIFFs from stale segmentation data.
