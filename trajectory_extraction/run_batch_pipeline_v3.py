#!/usr/bin/env python3
"""Run the production TIFF-to-cleaned-trajectory workflow for many cell crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import tifffile


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')


HERE = Path(__file__).parent.resolve()
SINGLE_CELL_RUNNER = HERE / 'run_full_pipeline_v3.py'


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def is_cell_crop(path: Path) -> tuple[bool, str]:
    """Accept only three/four-channel ImageJ crop TIFFs written by save_crops.py."""
    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            shape = tuple(int(value) for value in series.shape)
    except Exception as exc:
        return False, f'unreadable TIFF: {exc}'
    if axes not in ('TCYX', 'TZCYX'):
        return False, f'axes={axes}, expected TCYX or TZCYX'
    if shape[axes.index('C')] not in (3, 4):
        return False, f'shape={shape}, expected 3 or 4 channels'
    return True, f'axes={axes}, shape={shape}'


def completion_matches(analysis_dir: Path, args) -> bool:
    manifest_path = analysis_dir / 'trajectory_run_manifest.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            options = manifest.get('options', {})
            return (
                manifest.get('status') == 'complete'
                and options.get('reference_channel') == args.reference_channel
                and int(options.get('matlab_workers', -1)) == args.matlab_workers
                and bool(options.get('matlab_save_filter_images'))
                == args.matlab_save_filter_images
            )
        except (OSError, ValueError, TypeError):
            return False

    # Runs made before manifests existed have unknown scientific/concurrency
    # options, so the production batch wrapper deliberately does not skip them.
    return False


def tail_log(analysis_dir: Path, line_count: int = 20) -> str:
    log_path = analysis_dir / 'log_trajectory_v3.txt'
    if not log_path.is_file():
        return 'canonical log was not created'
    lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-line_count:])


def run_cell(crop: Path, args) -> dict:
    analysis_dir = crop.with_suffix('')
    if args.resume and completion_matches(analysis_dir, args):
        return {
            'crop': str(crop),
            'analysis_dir': str(analysis_dir),
            'status': 'skipped_complete',
            'started_at': '',
            'finished_at': now(),
            'duration_s': 0.0,
            'exit_code': 0,
            'error_tail': '',
        }

    command = [
        sys.executable,
        str(SINGLE_CELL_RUNNER),
        str(crop),
        '--fiji-bin',
        args.fiji_bin,
        '--reference-channel',
        args.reference_channel,
        '--matlab-bin',
        args.matlab_bin,
        '--matlab-workers',
        str(args.matlab_workers),
    ]
    if args.matlab_save_filter_images:
        command.append('--matlab-save-filter-images')

    started_at = now()
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
        )
        exit_code = int(result.returncode)
        status = 'complete' if exit_code == 0 else 'failed'
        error_tail = '' if exit_code == 0 else tail_log(analysis_dir)
    except Exception as exc:
        exit_code = -1
        status = 'failed_to_start'
        error_tail = repr(exc)

    return {
        'crop': str(crop),
        'analysis_dir': str(analysis_dir),
        'status': status,
        'started_at': started_at,
        'finished_at': now(),
        'duration_s': round(time.perf_counter() - started, 1),
        'exit_code': exit_code,
        'error_tail': error_tail,
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        'crop', 'analysis_dir', 'status', 'started_at', 'finished_at',
        'duration_s', 'exit_code', 'error_tail',
    ]
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Process all valid TCYX/TZCYX cell-crop TIFFs in one FOV directory. '
            'Mask/preview TIFFs are rejected by metadata inspection.'
        )
    )
    parser.add_argument('crop_dir', type=Path)
    parser.add_argument(
        '--crop-glob',
        default='*.tif',
        help='Non-recursive filename glob inside crop_dir (default: *.tif).',
    )
    parser.add_argument('--fiji-bin', default='fiji')
    parser.add_argument('--matlab-bin', default='matlab')
    parser.add_argument(
        '--reference-channel', choices=('green', 'red', 'purple'), default='green'
    )
    parser.add_argument(
        '--cell-workers',
        type=int,
        default=1,
        help='Cell pipelines run concurrently (default: 1).',
    )
    parser.add_argument(
        '--matlab-workers',
        type=int,
        choices=(1, 2, 3),
        default=1,
        help='MATLAB processes per active cell (default: 1).',
    )
    parser.add_argument('--matlab-save-filter-images', action='store_true')
    parser.add_argument(
        '--resume', action=argparse.BooleanOptionalAction, default=True,
        help='Skip compatible completed analyses (default: --resume).',
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.crop_dir = args.crop_dir.resolve()
    if not args.crop_dir.is_dir():
        raise SystemExit(f'ERROR: crop directory not found: {args.crop_dir}')
    if args.cell_workers < 1:
        raise SystemExit('ERROR: --cell-workers must be at least 1')

    candidates = sorted(path for path in args.crop_dir.glob(args.crop_glob) if path.is_file())
    crops = []
    print(f'Candidate TIFFs: {len(candidates)}')
    for path in candidates:
        accepted, reason = is_cell_crop(path)
        marker = 'ACCEPT' if accepted else 'SKIP'
        print(f'  [{marker}] {path.name}: {reason}')
        if accepted:
            crops.append(path)
    if not crops:
        raise SystemExit('ERROR: no valid TCYX 3/4-channel cell crop TIFFs found')

    total_matlab = args.cell_workers * args.matlab_workers
    print(f'Valid cell crops          : {len(crops)}')
    print(f'Concurrent cell pipelines: {args.cell_workers}')
    print(f'MATLAB workers per cell  : {args.matlab_workers}')
    print(f'Maximum MATLAB processes : {total_matlab}')
    print('GPU note: MATLAB SPT is CPU-based; CUDA only accelerates ND2 nucleus segmentation.')
    if args.dry_run:
        print('Dry run complete; no cell pipeline was started.')
        return

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.cell_workers) as executor:
        future_to_crop = {executor.submit(run_cell, crop, args): crop for crop in crops}
        pending = set(future_to_crop)
        while pending:
            completed, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            if not completed:
                print(
                    f'Batch still running: {len(pending)} cell(s) pending, '
                    f'{time.perf_counter() - started:.0f} s elapsed',
                    flush=True,
                )
                continue
            for future in completed:
                row = future.result()
                results.append(row)
                print(
                    f"[{row['status']}] {Path(row['crop']).name} "
                    f"({row['duration_s']} s)",
                    flush=True,
                )
                write_summary(args.crop_dir / 'trajectory_batch_summary.csv', results)

    order = {str(path): index for index, path in enumerate(crops)}
    results.sort(key=lambda row: order[row['crop']])
    summary_path = args.crop_dir / 'trajectory_batch_summary.csv'
    write_summary(summary_path, results)
    failures = [row for row in results if row['status'].startswith('failed')]
    print(f'Batch elapsed: {time.perf_counter() - started:.1f} s')
    print(f'Summary      : {summary_path}')
    print(f'Success/skip : {len(results) - len(failures)}/{len(results)}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
