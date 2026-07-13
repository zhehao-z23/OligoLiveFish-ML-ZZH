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
    - Python: install ../../requirements_tif_to_traj.txt
    - MATLAB must be on your PATH  (test with: matlab -batch "disp('ok')")
"""

import argparse
import sys
import re
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import scipy.io as sio
from pathlib import Path
from PIL import Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent.resolve()
MATLAB_DEPS  = HERE / 'matlab_deps'   # bundled .m files
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
# Step 1 — Run spt_batch.m on channel groups
# ═══════════════════════════════════════════════════════════════════════════════

def matlab_quote(value: Path | str) -> str:
    """Escape a path/string for a single-quoted MATLAB character vector."""
    return str(value).replace("'", "''")


def partition_channels(channels: list[str], worker_count: int) -> list[list[str]]:
    """Split three channels across MATLAB processes without starting extras."""
    groups = [[] for _ in range(min(worker_count, len(channels)))]
    for index, channel in enumerate(channels):
        groups[index % len(groups)].append(channel)
    return groups


def matlab_command(
    channels: list[str],
    tifs: dict[str, Path],
    frame_rate: float,
    pixl_um: float,
    out_dir: Path,
    save_filter_images: bool,
) -> str:
    keep_filters = 'true' if save_filter_images else 'false'
    calls = '; '.join(
        f"spt_batch('{matlab_quote(tifs[channel])}', {frame_rate}, {pixl_um}, "
        f"'{matlab_quote(out_dir)}', {keep_filters})"
        for channel in channels
    )
    return (
        f"addpath('{matlab_quote(MATLAB_DEPS)}'); "
        f"addpath('{matlab_quote(HERE)}'); {calls}"
    )


def run_matlab_group(
    channels: list[str],
    tifs: dict[str, Path],
    frame_rate: float,
    pixl_um: float,
    out_dir: Path,
    matlab_bin: str,
    save_filter_images: bool,
) -> tuple[list[str], int, str, float]:
    started = time.perf_counter()
    command = matlab_command(
        channels, tifs, frame_rate, pixl_um, out_dir, save_filter_images
    )
    result = subprocess.run(
        [matlab_bin, '-batch', command],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    output = '\n'.join(part for part in (result.stdout.rstrip(), result.stderr.rstrip()) if part)
    return channels, result.returncode, output, time.perf_counter() - started


def run_spt(
    tifs: dict[str, Path],
    frame_rate: float,
    pixl_um: float,
    out_dir: Path,
    matlab_bin: str,
    matlab_workers: int,
    save_filter_images: bool,
) -> None:
    channels = list(tifs)
    groups = partition_channels(channels, matlab_workers)
    print(f'  MATLAB processes: {len(groups)}')
    print(f'  Channel groups  : {groups}')
    print(f'  Save filter imgs: {save_filter_images}')
    for group in groups:
        names = ', '.join(tifs[channel].name for channel in group)
        print(f'  Queued MATLAB group [{", ".join(group)}]: {names}')

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        pending = {
            executor.submit(
                run_matlab_group,
                group,
                tifs,
                frame_rate,
                pixl_um,
                out_dir,
                matlab_bin,
                save_filter_images,
            )
            for group in groups
        }
        while pending:
            completed, pending = wait(
                pending, timeout=60, return_when=FIRST_COMPLETED
            )
            if not completed:
                print(
                    f'  MATLAB still running: {len(pending)} process(es), '
                    f'{time.perf_counter() - started:.0f} s elapsed',
                    flush=True,
                )
                continue
            for future in completed:
                results.append(future.result())

    failures = []
    for group, returncode, output, elapsed in results:
        label = ','.join(group)
        print(f'\n  --- MATLAB group {label} ({elapsed:.1f} s) ---')
        if output:
            print(output)
        if returncode != 0:
            failures.append((label, returncode))

    print(f'  MATLAB SPT elapsed: {time.perf_counter() - started:.1f} s')
    if failures:
        details = ', '.join(f'{label}=exit {code}' for label, code in failures)
        raise RuntimeError(f'MATLAB SPT failed: {details}')


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Export .mat trajectories to CSV
# ═══════════════════════════════════════════════════════════════════════════════

def export_trajectories(tifs: dict, out_dir: Path):
    csv_dir = out_dir / 'matlab_trajectory'
    csv_dir.mkdir(exist_ok=True)

    # A rerun may detect fewer candidates. Remove only Stage-2 candidate CSVs so
    # stale files cannot be matched as if they belonged to the new run.
    for prefix in _CHANNEL_PREFIX.values():
        for stale in csv_dir.glob(f'{prefix}_m2DGaussian_traj*.csv'):
            stale.unlink()

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MATLAB SPT and export candidate trajectories.'
    )
    parser.add_argument('input_dir', type=Path)
    parser.add_argument(
        '--matlab-bin',
        default='matlab',
        help='MATLAB executable or command (default: matlab from PATH).',
    )
    parser.add_argument(
        '--matlab-workers',
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            'Concurrent MATLAB processes. Default 1 handles all channels in one '
            'session; 2 or 3 can reduce wall time but require more RAM/CPU and '
            'sufficient MATLAB licensing.'
        ),
    )
    parser.add_argument(
        '--matlab-save-filter-images',
        action='store_true',
        help='Retain full filtered image stacks in MAT files (large; not needed for final CSVs).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
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
    print('Step 1: Running spt_batch.m on channel group(s)')
    print('=' * 60)
    run_spt(
        tifs,
        frame_rate,
        pixl_um,
        out_dir,
        args.matlab_bin,
        args.matlab_workers,
        args.matlab_save_filter_images,
    )

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
