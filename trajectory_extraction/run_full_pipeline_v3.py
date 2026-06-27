#!/usr/bin/env python3
"""
run_full_pipeline_v3.py — Single-nucleus trajectory extraction runner.

Runtime helper scripts and MATLAB dependencies are bundled in internal/.
No external SPT installation or hardcoded paths required.

Default steps:
  0. Fiji preprocessing from one cropped TIFF
  1. Detect green loci and output reference trajectories
  2. Run MATLAB 2D Gaussian SPT using bundled MATLAB dependencies
  3. Match MATLAB tracks to reference tracks and save cleaned trajectories

Usage:
    python3 run_full_pipeline_v3.py <cropped_tif>
    python3 run_full_pipeline_v3.py --no-fiji <try_analysis_dir>

Example:
    python3 run_full_pipeline_v3.py "/path/to/example_cropped.tif"
    python3 run_full_pipeline_v3.py --no-fiji /path/to/FOV5_analyzed/try_analysis
"""

import argparse
import sys
import subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
INTERNAL = HERE / 'internal'

V212_SCRIPT  = INTERNAL / 'auto_roi_for_published_v2.12.py'
SPT_SCRIPT   = INTERNAL / 'run_pipeline_v3.py'
MATCH_SCRIPT = INTERNAL / 'match_m2DGaussian_to_reference.py'
FIJI_MACRO   = INTERNAL / 'headless_Macro_first_steps_for_published.ijm'


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
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print(f"\n[ERROR] Command not found: {cmd[0]}")
        sys.exit(127)
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    if proc.returncode != 0:
        print(f"\n[ERROR] Command exited with code {proc.returncode} — aborting pipeline.")
        sys.exit(proc.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Oligo-LiveFISH trajectory extraction workflow.'
    )
    parser.add_argument(
        'input_path',
        help='Cropped input TIFF, or an already-preprocessed analysis directory with --no-fiji.'
    )
    parser.add_argument(
        '--no-fiji',
        action='store_true',
        help='Skip Fiji preprocessing and run trajectory extraction on an existing analysis directory.'
    )
    parser.add_argument(
        '--fiji-bin',
        default='fiji',
        help='Fiji/ImageJ executable to use for preprocessing (default: fiji).'
    )
    return parser.parse_args()


def analysis_dir_for_tif(tif_path: Path) -> Path:
    return tif_path.with_suffix('')


def prepare_analysis_dir(args) -> Path:
    input_path = Path(args.input_path).resolve()

    if args.no_fiji:
        if not input_path.is_dir():
            print(f"ERROR: --no-fiji expects an existing analysis directory: {input_path}")
            sys.exit(1)
        return input_path

    if not input_path.is_file():
        print(f"ERROR: expected a cropped .tif input file: {input_path}")
        print("       Use --no-fiji to run on a preprocessed analysis directory.")
        sys.exit(1)
    if input_path.suffix.lower() not in ('.tif', '.tiff'):
        print(f"ERROR: expected a .tif/.tiff input file: {input_path}")
        sys.exit(1)
    if not FIJI_MACRO.is_file():
        print(f"ERROR: Fiji macro not found: {FIJI_MACRO}")
        sys.exit(1)

    analysis_dir = analysis_dir_for_tif(input_path)
    analysis_dir.mkdir(exist_ok=True)
    return analysis_dir


def main():
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    analysis_dir = prepare_analysis_dir(args)

    log_path = analysis_dir / 'log_trajectory_v3.txt'
    with open(log_path, 'w') as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print(f"Pipeline started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Analysis dir     : {analysis_dir}")
            print(f"Log file         : {log_path}")

            if not args.no_fiji:
                print(f"Input TIFF       : {input_path}")
                print(f"Fiji macro       : {FIJI_MACRO}")
                run([
                    args.fiji_bin,
                    '--headless',
                    '-macro',
                    str(FIJI_MACRO),
                    str(input_path),
                ])

            nucleus_files = sorted(analysis_dir.glob('*_Nucleus.tif'))
            if not nucleus_files:
                print(f"ERROR: no *_Nucleus.tif found in {analysis_dir}")
                sys.exit(1)
            if len(nucleus_files) > 1:
                print(f"WARNING: multiple *_Nucleus.tif found; using {nucleus_files[0].name}")
            nucleus_path = nucleus_files[0]
            print(f"Nucleus file     : {nucleus_path.name}")

            # Step 1 — reference trajectories
            run(['python3', str(V212_SCRIPT), str(nucleus_path)])

            # Step 2 — MATLAB SPT trajectories (uses bundled internal/matlab_deps/)
            run(['python3', str(SPT_SCRIPT), str(analysis_dir)])

            # Step 3 — match MATLAB tracks to reference tracks
            run(['python3', str(MATCH_SCRIPT), str(analysis_dir)])

            print(f"\n{'═'*70}")
            print(f"Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'═'*70}")

        finally:
            sys.stdout = original_stdout

    print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
