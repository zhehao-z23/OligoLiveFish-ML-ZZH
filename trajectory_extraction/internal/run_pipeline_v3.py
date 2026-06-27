#!/usr/bin/env python3
"""
run_pipeline_v3.py — Run SPT directly on full-field channel TIFFs, no ROI segmentation.

Standalone version: all MATLAB dependencies are bundled in matlab_deps/ next to
this script. No external SPT installation required.

Usage:
    python3 run_pipeline_v3.py /path/to/input_dir/

The input_dir must contain:
    - <stem>_green.tif
    - <stem>_red.tif
    - <stem>_purple.tif

Outputs (all inside input_dir/matlab_result/):
    - <stem>_green.mat, <stem>_red.mat, <stem>_purple.mat
    - matlab_trajectory/G_m2DGaussian_traj1.csv, R_..., P_..., etc.

Requirements:
    - Python: Pillow, numpy, scipy  (pip install -r requirements.txt)
    - MATLAB must be on your PATH  (test with: matlab -batch "disp('ok')")
"""

import sys
import re
import subprocess
import scipy.io as sio
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent.resolve()
MATLAB_DEPS  = HERE / 'matlab_deps'   # bundled .m files
MATLAB_BIN   = 'matlab'               # resolved from PATH

_CHANNEL_PREFIX = {'green': 'G', 'red': 'R', 'purple': 'P'}


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════════

def read_tiff_metadata(path: Path) -> tuple:
    """Read frame_rate (Hz) and pixl_um (µm/px) from a TIFF's ImageJ metadata."""
    img  = Image.open(str(path))
    tags = img.tag_v2

    img_desc = tags.get(270, '')
    if isinstance(img_desc, bytes):
        img_desc = img_desc.decode('utf-8', errors='replace')
    elif isinstance(img_desc, tuple):
        img_desc = img_desc[0]
    m = re.search(r'finterval=([0-9.eE+\-]+)', img_desc)
    if not m:
        raise ValueError(f'finterval not found in ImageDescription of {path}')
    finterval  = float(m.group(1))
    frame_rate = 1.0 / finterval

    x_res_raw = tags.get(282)
    if x_res_raw is None:
        raise ValueError(f'XResolution tag not found in {path}')
    if isinstance(x_res_raw, tuple) and len(x_res_raw) == 2:
        x_res = x_res_raw[0] / x_res_raw[1]
    else:
        x_res = float(x_res_raw)
    pixl_um = 1.0 / x_res

    print(f'  finterval = {finterval:.4f} s  →  frame_rate = {frame_rate:.4f} Hz')
    print(f'  XResolution = {x_res:.4f} px/µm  →  pixl = {pixl_um:.4f} µm/px')
    return frame_rate, pixl_um


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Run spt_batch.m on each channel TIFF
# ═══════════════════════════════════════════════════════════════════════════════

def run_spt(tifs: dict, frame_rate: float, pixl_um: float, out_dir: Path):
    for ch, tif in tifs.items():
        print(f'\n  Running spt_batch on {tif.name} ...')
        matlab_cmd = (
            f"addpath('{MATLAB_DEPS}'); "
            f"addpath('{HERE}'); "
            f"spt_batch('{tif}', {frame_rate}, {pixl_um}, '{out_dir}')"
        )
        result = subprocess.run(
            [MATLAB_BIN, '-batch', matlab_cmd],
            input='y\n', capture_output=True, text=True
        )
        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip())
        if result.returncode != 0:
            print(f'  [ERROR] spt_batch failed for {tif.name} '
                  f'(exit code {result.returncode})')


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Export .mat trajectories to CSV
# ═══════════════════════════════════════════════════════════════════════════════

def export_trajectories(tifs: dict, out_dir: Path):
    csv_dir = out_dir / 'matlab_trajectory'
    csv_dir.mkdir(exist_ok=True)

    for ch, tif in tifs.items():
        prefix   = _CHANNEL_PREFIX[ch]
        mat_path = out_dir / (tif.stem + '.mat')
        if not mat_path.exists():
            print(f'  [SKIP] {mat_path.name} not found')
            continue

        mat     = sio.loadmat(str(mat_path))
        traj    = mat['traj']
        sptpara = mat['sptpara'][0, 0]
        pixl_nm = float(sptpara['pixl'].flat[0]) * 1000
        n_traj  = traj.shape[1]

        for i in range(n_traj):
            pos = traj[0, i]['pos']
            if pos.shape == (1, 1):
                pos = pos[0, 0]
            frames = pos[:, 2].astype(int)
            x_nm   = pos[:, 0] * pixl_nm
            y_nm   = pos[:, 1] * pixl_nm

            csv_name = f'{prefix}_m2DGaussian_traj{i+1}.csv'
            out_file = csv_dir / csv_name
            with open(out_file, 'w') as f:
                f.write('frame,x_nm,y_nm\n')
                for frame, x, y in zip(frames, x_nm, y_nm):
                    f.write(f'{frame},{x:.2f},{y:.2f}\n')
            print(f'    {out_file.relative_to(out_dir.parent)}')

        print(f'  {ch}: {n_traj} trajectory file(s) exported')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) != 2:
        print('Usage: python3 run_pipeline_v3.py /path/to/input_dir/')
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    if not input_dir.is_dir():
        print(f'Error: not a directory: {input_dir}')
        sys.exit(1)

    # Find channel TIFFs
    green_tifs = sorted(input_dir.glob('*_green.tif'))
    if not green_tifs:
        print(f'Error: no *_green.tif found in {input_dir}')
        sys.exit(1)
    if len(green_tifs) > 1:
        print(f'Warning: multiple *_green.tif found; using {green_tifs[0].name}')
    stem = green_tifs[0].name[:-len('_green.tif')]

    tifs = {}
    for ch in ('green', 'red', 'purple'):
        p = input_dir / f'{stem}_{ch}.tif'
        if not p.exists():
            print(f'Error: {p.name} not found in {input_dir}')
            sys.exit(1)
        tifs[ch] = p

    # Create output directory
    out_dir = input_dir / 'matlab_result'
    out_dir.mkdir(exist_ok=True)

    # Read metadata
    print('Reading TIFF metadata...')
    frame_rate, pixl_um = read_tiff_metadata(tifs['green'])

    # Step 1 — SPT
    print()
    print('=' * 60)
    print('Step 1: Running spt_batch.m on each channel')
    print('=' * 60)
    run_spt(tifs, frame_rate, pixl_um, out_dir)

    # Step 2 — Export
    print()
    print('=' * 60)
    print('Step 2: Exporting trajectories to CSV')
    print('=' * 60)
    export_trajectories(tifs, out_dir)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
