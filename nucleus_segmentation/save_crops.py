"""
save_crops.py — read JSON files produced by crop_nuclei_sam.py and save TIFFs.

Usage:
    conda run -n base python "code (being modified)/save_crops.py" \
        "data for analysis/FOV (.nd2 files)"

Finds all *_crops.json files under the given directory tree, then for each crop:
  1. Loads the original .nd2 → (T, Z, C, Y, X) uint16 via axis normalization
  2. Reads per-channel LUT colors and physical metadata from nd2
  3. Slices the bbox, zeros suppression pixels
  4. Saves as an ImageJ composite hyperstack TIFF with per-channel LUTs
     so Fiji opens with the same colors and calibration as the original nd2
"""

import re
import sys
import json
import argparse
import numpy as np
import tifffile
from pathlib import Path
from nd2 import ND2File


"""
_parse_exposures() parses per-channel exposure times from the nd2 text_info.
Inputs:  text_info — nd2 text_info dict; num_channels — number of channels
Outputs: list of length num_channels with exposure in ms (float) or None per channel
"""
def _parse_exposures(text_info: dict, num_channels: int) -> list:
    desc = text_info.get('description', '')
    exposures = [None] * num_channels
    # Split on "Plane #N:" boundaries; first element is pre-plane header text
    blocks = re.split(r'Plane #\d+:', desc)
    for i, block in enumerate(blocks[1:]):
        if i >= num_channels:
            break
        m = re.search(r'Exposure:\s*([0-9.]+)\s*(ms|s)\b', block)
        if m:
            val = float(m.group(1))
            if m.group(2) == 's':
                val *= 1000.0
            exposures[i] = val
    return exposures


"""
load_fov_with_metadata() loads an nd2 plus all metadata needed for TIFF export.
Inputs:  nd2_path — path to the .nd2 file
Outputs: (fov, meta) where fov is (T, Z, C, Y, X) uint16 and meta is the dict below.

Metadata dict contains:
      px_um_x/y/z    — pixel size in µm
      finterval_s    — frame interval in seconds (periodMs → periodDiff.avg fallback)
      chan_names     — list of channel name strings
      luts           — list of (3, 256) uint8 linear LUT arrays (from nd2 colors)
      display_ranges — flat tuple (min0, max0, min1, max1, ...) from nd2
                       componentMinima/Maxima, falling back to full-FOV percentiles
      emission_nm    — list of emission wavelengths in nm (or None)
      exposure_ms    — list of exposure times in ms parsed from text_info (or None)
      objective_name — string or None
      objective_na   — float or None
      magnification  — float or None
      modality       — string e.g. 'fluorescence' or None
"""
def load_fov_with_metadata(nd2_path: Path):
    with ND2File(nd2_path) as f:
        arr = f.asarray()
        keys = list(f.sizes.keys())
        channels = f.metadata.channels
        text_info = f.text_info or {}
        acquisition_date = text_info.get('date')

        # Physical pixel size in microns
        try:
            vx = f.voxel_size()
            px_um_x, px_um_y, px_um_z = float(vx.x), float(vx.y), float(vx.z)
        except Exception:
            px_um_x = px_um_y = px_um_z = 1.0

        # Frame interval: level 1 = periodMs (requested), level 2 = periodDiff.avg (measured)
        # periodMs and periodDiff.avg are both in ms (confirmed: periodDiff.avg ≈ 24426 ms
        # for ~24.4 s/frame experiments). Values > 1000 are treated as ms → divide by 1000.
        finterval_s = None
        try:
            for loop in f.experiment:
                if getattr(loop, 'type', '').lower().startswith('time'):
                    period_ms = getattr(loop.parameters, 'periodMs', None) \
                                or getattr(loop.parameters, 'period', None)
                    if period_ms:
                        val = float(period_ms)
                        finterval_s = val / 1000.0 if val > 1000 else val
                        break
                    avg = getattr(
                        getattr(loop.parameters, 'periodDiff', None), 'avg', None)
                    if avg and avg > 0:
                        val = float(avg)
                        finterval_s = val / 1000.0 if val > 1000 else val
                        print(f"  periodDiff.avg: {val} → finterval_s = {finterval_s:.4f} s")
                        break
        except Exception:
            pass

        # Per-channel nd2 display ranges (componentMinima/Maxima — often 0.0)
        nd2_ranges = []
        for ch in channels:
            try:
                lo = ch.volume.componentMinima[0]
                hi = ch.volume.componentMaxima[0]
                nd2_ranges.append((float(lo), float(hi)) if (lo > 0 or hi > 0) else None)
            except Exception:
                nd2_ranges.append(None)

        # Channel names and emission wavelengths
        chan_names = [str(ch.channel.name) if ch.channel.name else f"C{i}"
                      for i, ch in enumerate(channels)]
        emission_nm   = [getattr(ch.channel, 'emissionLambdaNm',   None) for ch in channels]
        excitation_nm = [getattr(ch.channel, 'excitationLambdaNm', None) for ch in channels]

        # Objective + modality (same scope settings for all channels — read from first)
        try:
            mic = channels[0].microscope
            objective_name = getattr(mic, 'objectiveName', None)
            objective_na   = getattr(mic, 'objectiveNumericalAperture', None)
            magnification  = getattr(mic, 'objectiveMagnification', None)
            modality       = (getattr(mic, 'modalityFlags', None) or [None])[0]
        except Exception:
            objective_name = objective_na = magnification = modality = None

        # Exposure per channel parsed from text_info description
        exposure_ms   = _parse_exposures(text_info, len(channels))

        # LUTs — linear ramp from channel color; nd2 stores no non-linear LUT
        ramp = np.arange(256, dtype=np.float32) / 255.0
        luts = []
        for ch in channels:
            try:
                col = ch.channel.color
                r, g, b = int(col.r), int(col.g), int(col.b)
            except Exception:
                r, g, b = 255, 255, 255
            lut = np.zeros((3, 256), dtype=np.uint8)
            lut[0] = (ramp * r).astype(np.uint8)
            lut[1] = (ramp * g).astype(np.uint8)
            lut[2] = (ramp * b).astype(np.uint8)
            luts.append(lut)

    # --- axis normalization (outside with-block; arr is fully loaded) ---
    axes = [k.upper() for k in keys]
    print(f"  nd2 axes string: {''.join(axes)}, raw shape: {arr.shape}")

    for ax in ('T', 'Z', 'C'):
        if ax not in axes:
            arr = np.expand_dims(arr, axis=0)
            axes.insert(0, ax)
            print(f"  Reinserted size-1 '{ax}' axis (squeezed out by nd2 library)")

    target = ['T', 'Z', 'C', 'Y', 'X']
    order = [axes.index(a) for a in target]
    fov = np.transpose(arr, order).astype(np.uint16)
    T, Z, C, H, W = fov.shape
    print(f"  Final shape TZCYX: {fov.shape}")

    # Pad channel lists if nd2 had fewer entries than C (shouldn't happen, but safe)
    white_lut = np.stack([np.arange(256, dtype=np.uint8)] * 3)
    while len(chan_names) < C:
        i = len(chan_names)
        chan_names.append(f"C{i}")
        emission_nm.append(None)
        excitation_nm.append(None)
        exposure_ms.append(None)
        nd2_ranges.append(None)
        luts.append(white_lut.copy())

    # Display ranges: prefer nd2 stored values; fall back to full-FOV percentiles.
    # Percentiles are computed on the FOV (not the suppressed crop) so the range
    # matches what Fiji shows when opening the original nd2.
    display_ranges = []
    for c in range(C):
        if nd2_ranges[c] is not None:
            display_ranges.append(nd2_ranges[c])
        else:
            plane = fov[:, :, c].astype(np.float32)
            nz = plane[plane > 0]
            if nz.size == 0:
                display_ranges.append((0.0, 1.0))
            else:
                lo = float(np.percentile(nz, 1))
                hi = float(np.percentile(nz, 99))
                display_ranges.append((lo, hi))

    # Flat tuple for ImageJ Ranges key: (min0, max0, min1, max1, ...)
    ranges_flat = tuple(v for lo, hi in display_ranges for v in (lo, hi))

    return fov, {
        'px_um_x':          px_um_x,
        'px_um_y':          px_um_y,
        'px_um_z':          px_um_z,
        'finterval_s':      finterval_s,
        'chan_names':       chan_names,
        'luts':             luts,
        'display_ranges':   ranges_flat,
        'emission_nm':      emission_nm,
        'excitation_nm':    excitation_nm,
        'exposure_ms':      exposure_ms,
        'objective_name':   objective_name,
        'objective_na':     objective_na,
        'magnification':    magnification,
        'modality':         modality,
        'acquisition_date': acquisition_date,
    }


"""
_write_sidecar() writes a <stem>_<idx>_metadata.json next to each crop TIFF.
Inputs:  path — crop TIFF path; nd2_path — source nd2; stem, idx — crop identity;
         crop_shape — (T, Z, C, H, W); bbox — (r0, r1, c0, c1); meta — metadata dict
Outputs: none (writes the JSON sidecar)

Captures the acquisition metadata that the ImageJ TIFF format has no standard field
for: emission/excitation wavelengths, exposure times, objective info, and per-channel
display ranges.
"""
def _write_sidecar(path: Path, nd2_path: Path, stem: str, idx: int,
                   crop_shape: tuple, bbox: tuple, meta: dict):
    T, Z, C, H, W = crop_shape
    r0, r1, c0, c1 = bbox

    channels = []
    for c in range(C):
        lo = meta['display_ranges'][c * 2]
        hi = meta['display_ranges'][c * 2 + 1]
        channels.append({
            'index':          c,
            'name':           meta['chan_names'][c],
            'emission_nm':    meta['emission_nm'][c],
            'excitation_nm':  meta['excitation_nm'][c],
            'exposure_ms':    meta['exposure_ms'][c],
            'display_min':    lo,
            'display_max':    hi,
        })

    sidecar = {
        'source_nd2':       str(nd2_path),
        'acquisition_date': meta['acquisition_date'],
        'stem':             stem,
        'crop_index':       idx,
        'crop_shape':       {'T': T, 'Z': Z, 'C': C, 'Y': H, 'X': W},
        'bbox':             {'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1},
        'pixel_size': {
            'x_um': meta['px_um_x'],
            'y_um': meta['px_um_y'],
            'z_um': meta['px_um_z'],
        },
        'time': {
            'finterval_s': meta['finterval_s'],
            'fps':         1.0 / meta['finterval_s'] if meta['finterval_s'] else None,
            'n_frames':    T,
        },
        'acquisition': {
            'objective':     meta['objective_name'],
            'na':            meta['objective_na'],
            'magnification': meta['magnification'],
            'modality':      meta['modality'],
        },
        'channels': channels,
    }

    sidecar_path = path.with_name(path.stem + '_metadata.json')
    with open(sidecar_path, 'w') as fh:
        json.dump(sidecar, fh, indent=2)


"""
process_json() writes the TIFF crops + metadata sidecars for one crops JSON.
Inputs:  json_path — path to a <stem>_crops.json written by crop_nuclei_sam.py
Outputs: number of TIFFs saved (also writes each TIFF + _metadata.json sidecar)

1. load the source nd2 + metadata referenced by the JSON
2. for each crop: slice the bbox, zero the suppression pixels, write an ImageJ
   composite hyperstack TIFF with per-channel LUTs + calibration
3. write a metadata sidecar per crop; read-back-verify the first crop
"""
def process_json(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)

    nd2_path = Path(data['nd2_path'])
    stem = data['stem']
    crops = data['crops']

    if not nd2_path.exists():
        print(f"  ERROR: nd2 not found: {nd2_path}", file=sys.stderr)
        return 0

    print(f"\n── {nd2_path.name} ──")
    fov, meta = load_fov_with_metadata(nd2_path)
    T, Z, C, H, W = fov.shape
    print(f"  Loaded: T={T} Z={Z} C={C} Y={H} X={W}")
    print(f"  Pixel size: {meta['px_um_x']:.4f} × {meta['px_um_y']:.4f} µm/px"
          f"  Z-step: {meta['px_um_z']:.4f} µm")
    print(f"  Frame interval: {meta['finterval_s']} s")
    print(f"  Channels: {meta['chan_names']}")
    print(f"  Emission (nm): {meta['emission_nm']}")
    print(f"  Exposure (ms): {meta['exposure_ms']}")
    print(f"  Objective: {meta['objective_name']}  NA={meta['objective_na']}")
    print(f"  Display ranges: {meta['display_ranges']}")

    # ImageJ resolution: pixels-per-µm (ResolutionUnit=NONE lets unit= key govern)
    xres = 1.0 / meta['px_um_x'] if meta['px_um_x'] else 1.0
    yres = 1.0 / meta['px_um_y'] if meta['px_um_y'] else 1.0

    out_dir = json_path.parent
    saved = 0

    for crop_info in crops:
        idx = crop_info['idx']
        r0, r1, c0, c1 = crop_info['bbox']
        sup_rows = np.array(crop_info['suppression_rows'], dtype=np.int32)
        sup_cols = np.array(crop_info['suppression_cols'], dtype=np.int32)

        # Slice spatial dims (Y, X) — preserve T and Z fully
        crop = fov[:, :, :, r0:r1, c0:c1].copy()
        if sup_rows.size > 0:
            crop[:, :, :, sup_rows, sup_cols] = 0

        # ImageJ Labels: one entry per T*Z*C plane, T-major → Z-major → C-major
        labels = [
            meta['chan_names'][c]
            for t in range(T)
            for z in range(Z)
            for c in range(C)
        ]

        ij_metadata = {
            'axes':   'TZCYX',
            'mode':   'color',
            'unit':   'um',
            'tunit':  's',
            'LUTs':   meta['luts'],
            'Labels': labels,
            'Ranges': meta['display_ranges'],
        }

        finterval = meta['finterval_s']
        if finterval is None:
            print("  WARNING: finterval_s not found; defaulting to 1.0 s")
            finterval = 1.0
        ij_metadata['finterval'] = finterval
        ij_metadata['fps'] = 1.0 / finterval

        # spacing is only meaningful when Z > 1; omit for max-projected (Z=1) crops
        if Z > 1:
            ij_metadata['spacing'] = meta['px_um_z']

        tif_path = out_dir / f"{stem}_{idx}.tif"
        tifffile.imwrite(
            tif_path,
            crop,
            imagej=True,
            photometric='minisblack',
            resolution=(xres, yres),
            metadata=ij_metadata,
        )
        _write_sidecar(tif_path, nd2_path, stem, idx,
                       crop.shape, (r0, r1, c0, c1), meta)
        saved += 1

        # Read-back verification on the first crop per nd2 file
        if saved == 1:
            with tifffile.TiffFile(tif_path) as tf:
                ij = tf.imagej_metadata or {}
                print(f"  [verify] axes={ij.get('axes')}  unit={ij.get('unit')}"
                      f"  tunit={ij.get('tunit')}")
                print(f"  [verify] finterval={ij.get('finterval')}, fps={ij.get('fps')}")
                print(f"  [verify] spacing="
                      f"{'<absent>' if 'spacing' not in ij else ij['spacing']}")
                print(f"  [verify] Ranges: {ij.get('Ranges')}")
                print(f"  [verify] LUTs present: {len(ij.get('LUTs', []))} channels")

    print(f"  ✓ {saved} TIFFs → {out_dir}/")
    return saved


"""
main() finds every crops JSON under the input directory and processes each one.
Inputs:  none (reads command-line arguments)
Outputs: none (recursively finds *_crops.json, calls process_json on each, prints a summary)
"""
def main():
    parser = argparse.ArgumentParser(
        description="Save TIFFs from crop JSON files produced by crop_nuclei_sam.py")
    parser.add_argument("input",
                        help="directory containing *_crops.json files (searched recursively)")
    args = parser.parse_args()

    root = Path(args.input)
    json_files = sorted(root.rglob("*_crops.json"))
    if not json_files:
        print(f"No *_crops.json files found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s):")
    for j in json_files:
        print(f"  {j}")

    total = 0
    for j in json_files:
        total += process_json(j)

    print(f"\nDone. {total} total TIFFs saved.")


if __name__ == "__main__":
    main()
