#!/usr/bin/env python3
"""
run_full_pipeline_v3.py — Single-nucleus trajectory extraction runner.

Runtime helper scripts and MATLAB dependencies are bundled in pipeline/.
No external SPT installation or hardcoded paths required.

Default steps:
  0. Fiji preprocessing from one cropped TIFF
  1. Detect loci in the selected reference channel and output reference trajectories
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
import json
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).parent.resolve()
PIPELINE = HERE / 'pipeline'

STAGE1_SCRIPT = PIPELINE / 'auto_roi_for_published_v2.13.py'
SPT_SCRIPT    = PIPELINE / 'run_pipeline_v3.py'
MATCH_SCRIPT  = PIPELINE / 'match_m2DGaussian_to_reference.py'
FIJI_MACRO    = PIPELINE / 'headless_Macro_first_steps_for_published.ijm'


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
    parser.add_argument(
        '--reference-channel',
        choices=('green', 'red', 'purple'),
        default='green',
        help='Stage 1 anchor channel; the other two channels are tracked as targets (default: green).',
    )
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
            'Concurrent MATLAB processes for the three locus channels. Default 1 '
            'reuses one MATLAB session; 2 or 3 can reduce wall time on machines '
            'with sufficient CPU, RAM, and MATLAB licensing.'
        ),
    )
    parser.add_argument(
        '--matlab-save-filter-images',
        action='store_true',
        help='Retain large filtered image stacks in Stage-2 MAT files.',
    )
    parser.add_argument('--skip-stage1', action='store_true', help='Reuse existing reference CSVs.')
    parser.add_argument('--skip-stage2', action='store_true', help='Reuse existing MATLAB candidate CSVs.')
    parser.add_argument('--skip-stage3', action='store_true', help='Do not rerun final matching.')
    return parser.parse_args()


def analysis_dir_for_tif(tif_path: Path) -> Path:
    return tif_path.with_suffix('')


def prepare_fiji_input_path(input_path: Path) -> tuple[Path, tuple[Path, Path] | None]:
    """Return an ASCII-only view of a TIFF path for the legacy Windows Fiji launcher.

    Fiji's native Windows launcher can corrupt non-ASCII command-line macro
    arguments before Java starts.  A temporary directory symlink lets Fiji see
    an ASCII path while all reads and writes still resolve to the original
    directory (for example, a Chinese-named folder on G:).
    """
    if os.name != 'nt' or str(input_path).isascii():
        return input_path, None
    if not input_path.name.isascii():
        raise RuntimeError(
            'Fiji on Windows requires an ASCII TIFF filename. The parent directory '
            'may contain non-ASCII characters and will be bridged automatically, '
            f'but rename the TIFF itself first: {input_path.name}'
        )

    bridge_root = Path(tempfile.mkdtemp(prefix='oligolivefish_fiji_'))
    bridge_dir = bridge_root / 'source'
    try:
        os.symlink(input_path.parent, bridge_dir, target_is_directory=True)
        bridged_input = bridge_dir / input_path.name
        if not bridged_input.is_file():
            raise FileNotFoundError(f'Fiji bridge cannot see input TIFF: {bridged_input}')
    except Exception:
        if bridge_dir.exists() or bridge_dir.is_symlink():
            os.unlink(bridge_dir)
        bridge_root.rmdir()
        raise

    print(f'Fiji path bridge : {bridged_input}')
    print('                   (resolves to the original input/output directory)')
    return bridged_input, (bridge_dir, bridge_root)


def cleanup_fiji_input_path(bridge: tuple[Path, Path] | None) -> None:
    if bridge is None:
        return
    bridge_dir, bridge_root = bridge
    if bridge_dir.exists() or bridge_dir.is_symlink():
        os.unlink(bridge_dir)
    if bridge_root.exists():
        bridge_root.rmdir()


def create_synthetic_nucleus_if_needed(analysis_dir: Path) -> Path | None:
    """Create the Stage-1 scaffold required for a three-colour acquisition."""
    if list(analysis_dir.glob('*_Nucleus.tif')):
        return None
    channel_paths = {
        name: sorted(analysis_dir.glob(f'*_{name}.tif'))
        for name in ('green', 'red', 'purple')
    }
    if any(len(paths) != 1 for paths in channel_paths.values()):
        return None

    import numpy as np
    import tifffile

    green_path = channel_paths['green'][0]
    with tifffile.TiffFile(green_path) as tif:
        series = tif.series[0]
        if series.axes != 'TYX':
            raise ValueError(
                f'Expected Fiji three-channel output axes TYX, got {series.axes}: {green_path}'
            )
        frames, height, width = (int(value) for value in series.shape)
        imagej = dict(tif.imagej_metadata or {})
        page0 = tif.pages[0]
        x_resolution = page0.tags['XResolution'].value
        y_resolution = page0.tags['YResolution'].value

    frame = np.full((height, width), 1000, dtype=np.uint16)
    ys = np.arange(8, max(8, height - 8), 4)
    xs = np.arange(8, max(8, width - 8), 4)
    if len(ys) and len(xs):
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        frame[yy.ravel(), xx.ravel()] = 0
    stack = np.repeat(frame[np.newaxis, :, :], frames, axis=0)

    stem = green_path.name[:-len('_green.tif')]
    nucleus_path = analysis_dir / f'{stem}_Nucleus.tif'
    metadata = {
        'axes': 'TYX',
        'unit': imagej.get('unit', 'um'),
        'tunit': imagej.get('tunit', 's'),
        'finterval': float(imagej.get('finterval', 1.0)),
        'loop': False,
    }
    tifffile.imwrite(
        nucleus_path,
        stack,
        imagej=True,
        metadata=metadata,
        resolution=(x_resolution, y_resolution),
    )
    print('Three-channel input: green/red/purple detected.')
    print(f'Synthetic nucleus : {nucleus_path.name} (Stage-1 scaffold only)')
    print('WARNING: Do not use this synthetic image for nucleus morphology or density features.')
    return nucleus_path


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
    manifest_path = analysis_dir / 'trajectory_run_manifest.json'
    manifest = {
        'status': 'running',
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'input_path': str(input_path),
        'analysis_dir': str(analysis_dir),
        'python_executable': sys.executable,
        'options': {
            'no_fiji': args.no_fiji,
            'fiji_bin': args.fiji_bin,
            'reference_channel': args.reference_channel,
            'matlab_bin': args.matlab_bin,
            'matlab_workers': args.matlab_workers,
            'matlab_save_filter_images': args.matlab_save_filter_images,
            'skip_stage1': args.skip_stage1,
            'skip_stage2': args.skip_stage2,
            'skip_stage3': args.skip_stage3,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    completed = False
    with open(log_path, 'w', encoding='utf-8') as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print(f"Pipeline started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Analysis dir     : {analysis_dir}")
            print(f"Log file         : {log_path}")

            if not args.no_fiji:
                print(f"Input TIFF       : {input_path}")
                print(f"Fiji macro       : {FIJI_MACRO}")
                try:
                    fiji_input, bridge = prepare_fiji_input_path(input_path)
                except Exception as exc:
                    print(f'ERROR: unable to prepare Fiji input path: {exc}')
                    sys.exit(1)
                try:
                    run([
                        args.fiji_bin,
                        '--headless',
                        '-macro',
                        str(FIJI_MACRO),
                        str(fiji_input),
                    ])
                finally:
                    cleanup_fiji_input_path(bridge)

                try:
                    create_synthetic_nucleus_if_needed(analysis_dir)
                except Exception as exc:
                    print(f'ERROR: unable to prepare three-channel nucleus scaffold: {exc}')
                    sys.exit(1)

            nucleus_files = sorted(analysis_dir.glob('*_Nucleus.tif'))
            if not nucleus_files:
                print(f"ERROR: no *_Nucleus.tif found in {analysis_dir}")
                sys.exit(1)
            if len(nucleus_files) > 1:
                print(f"WARNING: multiple *_Nucleus.tif found; using {nucleus_files[0].name}")
            nucleus_path = nucleus_files[0]
            print(f"Nucleus file     : {nucleus_path.name}")

            # Step 1 — reference trajectories
            if args.skip_stage1:
                if not list(analysis_dir.glob('*_traj_rela2wholeimg.csv')):
                    print('ERROR: --skip-stage1 requested but no reference CSVs exist.')
                    sys.exit(1)
                print('Skipping Stage 1; reusing existing reference CSVs.')
            else:
                run([
                    sys.executable,
                    str(STAGE1_SCRIPT),
                    str(nucleus_path),
                    '--reference-channel',
                    args.reference_channel,
                ])

            # Step 2 — MATLAB SPT trajectories (uses bundled internal/matlab_deps/)
            if args.skip_stage2:
                candidate_dir = analysis_dir / 'matlab_result' / 'matlab_trajectory'
                if not list(candidate_dir.glob('*_m2DGaussian_traj*.csv')):
                    print('ERROR: --skip-stage2 requested but no MATLAB candidate CSVs exist.')
                    sys.exit(1)
                print('Skipping Stage 2; reusing existing MATLAB candidate CSVs.')
            else:
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
            if args.skip_stage3:
                print('Skipping Stage 3.')
            else:
                run([sys.executable, str(MATCH_SCRIPT), str(analysis_dir)])

            print(f"\n{'═'*70}")
            print(f"Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'═'*70}")
            completed = True

        finally:
            sys.stdout = original_stdout
            manifest['status'] = 'complete' if completed else 'failed_or_interrupted'
            manifest['finished_at'] = datetime.now().isoformat(timespec='seconds')
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
