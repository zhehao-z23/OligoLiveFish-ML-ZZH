#!/usr/bin/env python3
"""
run_full_pipeline_v3.py — Standalone full analysis pipeline.

Standalone version: all MATLAB dependencies are bundled in matlab_deps/ next to
this script. No external SPT installation or hardcoded paths required.

Steps:
  1. auto_roi_for_published_v2.13.py — detect selected reference-channel loci, output reference trajectories
                                       (joint seeding for overlapping ROIs)
  2. run_pipeline_v3.py              — run MATLAB SPT using bundled matlab_deps/
  3. match_m2DGaussian_to_reference.py — match MATLAB tracks to reference tracks

Usage:
    python3 run_full_pipeline_v3.py <try_analysis_dir>

Example:
    python3 run_full_pipeline_v3.py /path/to/FOV5_analyzed/try_analysis
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).parent.resolve()

STAGE1_SCRIPT = HERE / 'auto_roi_for_published_v2.13.py'
SPT_SCRIPT   = HERE / 'run_pipeline_v3.py'
MATCH_SCRIPT = HERE / 'match_m2DGaussian_to_reference.py'


class Tee:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log    = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def fileno(self):
        return self._stream.fileno()


def run(cmd: list):
    print(f"\n{'═'*70}")
    print('Running: ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
    print(f"{'═'*70}")
    env = os.environ.copy()
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    if proc.returncode != 0:
        print(f"\n[ERROR] Command exited with code {proc.returncode} — aborting pipeline.")
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the standalone trajectory extraction pipeline.")
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument(
        "--reference-channel",
        choices=("green", "red", "purple"),
        default="green",
        help="Stage 1 anchor channel; the other two channels are targets (default: green).",
    )
    parser.add_argument('--matlab-bin', default='matlab')
    parser.add_argument('--matlab-workers', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--matlab-save-filter-images', action='store_true')
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    if not analysis_dir.is_dir():
        print(f"ERROR: not a directory: {analysis_dir}")
        sys.exit(1)

    log_path = analysis_dir / 'log_trajectory_v3.txt'
    with open(log_path, 'w', encoding='utf-8') as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print(f"Pipeline started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Analysis dir     : {analysis_dir}")
            print(f"Log file         : {log_path}")

            nucleus_files = sorted(analysis_dir.glob('*_Nucleus.tif'))
            if not nucleus_files:
                print(f"ERROR: no *_Nucleus.tif found in {analysis_dir}")
                sys.exit(1)
            if len(nucleus_files) > 1:
                print(f"WARNING: multiple *_Nucleus.tif found; using {nucleus_files[0].name}")
            nucleus_path = nucleus_files[0]
            print(f"Nucleus file     : {nucleus_path.name}")

            # Step 1 — reference trajectories
            run([
                sys.executable,
                str(STAGE1_SCRIPT),
                str(nucleus_path),
                "--reference-channel",
                args.reference_channel,
            ])

            # Step 2 — MATLAB SPT trajectories (uses bundled matlab_deps/)
            stage2_cmd = [
                sys.executable,
                str(SPT_SCRIPT),
                str(analysis_dir),
                '--matlab-bin',
                args.matlab_bin,
                '--matlab-workers',
                str(args.matlab_workers),
            ]
            if args.matlab_save_filter_images:
                stage2_cmd.append('--matlab-save-filter-images')
            run(stage2_cmd)

            # Step 3 — match MATLAB tracks to reference tracks
            run([sys.executable, str(MATCH_SCRIPT), str(analysis_dir)])

            print(f"\n{'═'*70}")
            print(f"Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'═'*70}")

        finally:
            sys.stdout = original_stdout

    print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
