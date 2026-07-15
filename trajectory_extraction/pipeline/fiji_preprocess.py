"""Shared Fiji preprocessing helpers for production trajectory runners."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIJI_MACRO = HERE / "headless_Macro_first_steps_for_published.ijm"


def analysis_dir_for_crop(crop_tiff: Path) -> Path:
    return crop_tiff.with_suffix("")


def prepare_fiji_input_path(input_path: Path) -> tuple[Path, tuple[Path, Path] | None]:
    """Give the legacy Windows Fiji launcher an ASCII view of a Unicode folder."""
    if os.name != "nt" or str(input_path).isascii():
        return input_path, None
    if not input_path.name.isascii():
        raise RuntimeError(
            "Fiji on Windows requires an ASCII TIFF filename. Its parent path may "
            f"contain Unicode and is bridged automatically; rename {input_path.name!r}."
        )
    bridge_root = Path(tempfile.mkdtemp(prefix="oligolivefish_fiji_"))
    bridge_dir = bridge_root / "source"
    try:
        os.symlink(input_path.parent, bridge_dir, target_is_directory=True)
        bridged_input = bridge_dir / input_path.name
        if not bridged_input.is_file():
            raise FileNotFoundError(bridged_input)
    except Exception:
        if bridge_dir.exists() or bridge_dir.is_symlink():
            os.unlink(bridge_dir)
        bridge_root.rmdir()
        raise
    return bridged_input, (bridge_dir, bridge_root)


def cleanup_fiji_input_path(bridge: tuple[Path, Path] | None) -> None:
    if bridge is None:
        return
    bridge_dir, bridge_root = bridge
    if bridge_dir.exists() or bridge_dir.is_symlink():
        os.unlink(bridge_dir)
    if bridge_root.exists():
        bridge_root.rmdir()


def run_fiji(crop_tiff: Path, fiji_bin: str) -> Path:
    if not FIJI_MACRO.is_file():
        raise FileNotFoundError(FIJI_MACRO)
    analysis_dir = analysis_dir_for_crop(crop_tiff)
    analysis_dir.mkdir(exist_ok=True)
    bridged_input, bridge = prepare_fiji_input_path(crop_tiff)
    try:
        command = [
            fiji_bin,
            "--headless",
            "-macro",
            str(FIJI_MACRO),
            str(bridged_input),
        ]
        print("Running Fiji: " + " ".join(f'\"{item}\"' for item in command), flush=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        process.wait()
        if process.returncode:
            raise RuntimeError(f"Fiji exited with code {process.returncode}")
    finally:
        cleanup_fiji_input_path(bridge)
    return analysis_dir


def create_synthetic_nucleus_if_needed(analysis_dir: Path) -> Path | None:
    """Create only the Stage-1 filename scaffold for a three-channel crop."""
    existing = sorted(analysis_dir.glob("*_Nucleus.tif"))
    if existing:
        return existing[0]

    import numpy as np
    import tifffile

    channels = {
        name: sorted(analysis_dir.glob(f"*_{name}.tif"))
        for name in ("green", "red", "purple")
    }
    if any(len(paths) != 1 for paths in channels.values()):
        return None
    green_path = channels["green"][0]
    with tifffile.TiffFile(green_path) as tif:
        series = tif.series[0]
        if series.axes != "TYX":
            raise ValueError(f"Expected Fiji TYX output, found {series.axes}: {green_path}")
        frames, height, width = (int(value) for value in series.shape)
        imagej = dict(tif.imagej_metadata or {})
        x_resolution = tif.pages[0].tags["XResolution"].value
        y_resolution = tif.pages[0].tags["YResolution"].value

    # This image is never used as the biological nucleus boundary in v4. The
    # drift-aligned micro-SAM mask is passed explicitly to Stage 1.
    stack = np.ones((frames, height, width), dtype=np.uint8)
    stem = green_path.name[: -len("_green.tif")]
    path = analysis_dir / f"{stem}_Nucleus.tif"
    tifffile.imwrite(
        path,
        stack,
        imagej=True,
        metadata={
            "axes": "TYX",
            "unit": imagej.get("unit", "um"),
            "tunit": imagej.get("tunit", "s"),
            "finterval": float(imagej.get("finterval", 1.0)),
            "loop": False,
        },
        resolution=(x_resolution, y_resolution),
    )
    print(f"Three-channel Stage-1 scaffold created: {path.name}")
    return path


def one_nucleus_tiff(analysis_dir: Path) -> Path:
    paths = sorted(analysis_dir.glob("*_Nucleus.tif"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_Nucleus.tif in {analysis_dir}, found {len(paths)}"
        )
    return paths[0]
