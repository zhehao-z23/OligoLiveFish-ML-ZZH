#!/usr/bin/env python3
"""Prepare a trajectory-analysis directory from three registered TYX TIFFs.

The trajectory pipeline expects files named ``*_green.tif``, ``*_red.tif``,
``*_purple.tif`` and ``*_Nucleus.tif`` in one directory.  When the acquisition
does not contain a nuclear channel, this tool creates the same full-frame
synthetic nucleus surrogate used by the crop batch runner.  The surrogate is
only an input scaffold for Stage 1 and must not be used for nuclear-feature
measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile


def rational_to_float(value) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def inspect_tiff(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        shape = tuple(int(v) for v in series.shape)
        axes = series.axes
        if axes != "TYX":
            raise ValueError(f"Expected a TYX TIFF, got axes={axes}, shape={shape}: {path}")
        imagej = dict(tf.imagej_metadata or {})
        page0 = tf.pages[0]
        if "XResolution" not in page0.tags:
            raise ValueError(f"XResolution is missing: {path}")
        xres = rational_to_float(page0.tags["XResolution"].value)
        yres = (
            rational_to_float(page0.tags["YResolution"].value)
            if "YResolution" in page0.tags
            else xres
        )
        finterval = imagej.get("finterval")
    return {
        "shape": shape,
        "axes": axes,
        "x_resolution_px_per_um": xres,
        "y_resolution_px_per_um": yres,
        "finterval_s": float(finterval) if finterval is not None else None,
    }


def synthetic_nucleus_stack(frames: int, height: int, width: int) -> np.ndarray:
    """Return a full-frame surrogate whose sparse holes keep Otsu well-defined."""
    frame = np.full((height, width), 1000, dtype=np.uint16)
    ys = np.arange(8, max(8, height - 8), 4)
    xs = np.arange(8, max(8, width - 8), 4)
    if len(ys) and len(xs):
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        frame[yy.ravel(), xx.ravel()] = 0
    return np.repeat(frame[np.newaxis, :, :], frames, axis=0)


def destination_paths(output_dir: Path, stem: str) -> dict[str, Path]:
    return {
        role: output_dir / f"{stem}_{role}.tif"
        for role in ("green", "red", "purple", "Nucleus")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare registered green/red/purple TIFFs for trajectory extraction."
    )
    parser.add_argument("--green", required=True, type=Path)
    parser.add_argument("--red", required=True, type=Path)
    parser.add_argument("--purple", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem", default="trajectory_input")
    parser.add_argument(
        "--nucleus",
        type=Path,
        default=None,
        help="Optional registered TYX nucleus TIFF. If omitted, create a synthetic surrogate.",
    )
    parser.add_argument(
        "--frame-interval-s",
        type=float,
        default=None,
        help="Override frame interval. By default it is read from the green TIFF.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = {"green": args.green, "red": args.red, "purple": args.purple}
    metadata = {role: inspect_tiff(path) for role, path in sources.items()}
    shapes = {info["shape"] for info in metadata.values()}
    if len(shapes) != 1:
        raise ValueError(f"Channel shapes differ: {metadata}")
    shape = next(iter(shapes))
    frames, height, width = shape

    green_meta = metadata["green"]
    for role, info in metadata.items():
        if not math.isclose(
            info["x_resolution_px_per_um"],
            green_meta["x_resolution_px_per_um"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        ) or not math.isclose(
            info["y_resolution_px_per_um"],
            green_meta["y_resolution_px_per_um"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Channel pixel scales differ; register/export all channels with the "
                f"same calibration before trajectory analysis. green={green_meta}, "
                f"{role}={info}"
            )

    finterval = args.frame_interval_s
    if finterval is None:
        finterval = green_meta["finterval_s"]
    if finterval is None or finterval <= 0:
        raise ValueError("Frame interval is missing; pass --frame-interval-s explicitly.")
    if args.frame_interval_s is None:
        for role, info in metadata.items():
            channel_interval = info["finterval_s"]
            if channel_interval is None or not math.isclose(
                channel_interval, finterval, rel_tol=1e-6, abs_tol=1e-9
            ):
                raise ValueError(
                    "Channel frame intervals differ or are missing; pass "
                    "--frame-interval-s only after confirming the acquisition timing. "
                    f"green={finterval}, {role}={channel_interval}"
                )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = destination_paths(output_dir, args.stem)
    manifest_path = output_dir / "input_manifest.json"
    outputs = [*destinations.values(), manifest_path]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Prepared outputs already exist; use --overwrite to replace them: "
            + ", ".join(str(p) for p in existing)
        )

    for role, source in sources.items():
        shutil.copy2(source, destinations[role])

    if args.nucleus is not None:
        nucleus_meta = inspect_tiff(args.nucleus)
        if nucleus_meta["shape"] != shape:
            raise ValueError(
                f"Nucleus shape {nucleus_meta['shape']} does not match channel shape {shape}."
            )
        shutil.copy2(args.nucleus, destinations["Nucleus"])
        synthetic_nucleus = False
    else:
        nucleus = synthetic_nucleus_stack(frames, height, width)
        tifffile.imwrite(
            destinations["Nucleus"],
            nucleus,
            imagej=True,
            photometric="minisblack",
            resolution=(
                green_meta["x_resolution_px_per_um"],
                green_meta["y_resolution_px_per_um"],
            ),
            metadata={
                "axes": "TYX",
                "unit": "micron",
                "tunit": "s",
                "finterval": finterval,
                "fps": 1.0 / finterval,
                "loop": False,
            },
        )
        synthetic_nucleus = True

    manifest = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "shape_tyx": list(shape),
        "finterval_s": finterval,
        "x_resolution_px_per_um": green_meta["x_resolution_px_per_um"],
        "pixel_size_nm": 1000.0 / green_meta["x_resolution_px_per_um"],
        "synthetic_nucleus": synthetic_nucleus,
        "warning": (
            "Synthetic nucleus is only a Stage-1 scaffold and must not be used "
            "for nuclear morphology or chromatin-density features."
            if synthetic_nucleus
            else ""
        ),
        "sources": {role: str(path.resolve()) for role, path in sources.items()},
        "outputs": {role: str(path) for role, path in destinations.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared analysis directory: {output_dir}")
    print(f"Shape: {shape} TYX")
    print(f"Frame interval: {finterval:.6f} s")
    print(
        "Pixel scale: "
        f"{green_meta['x_resolution_px_per_um']:.6f} px/um "
        f"({1000.0 / green_meta['x_resolution_px_per_um']:.6f} nm/px)"
    )
    print(f"Synthetic nucleus: {synthetic_nucleus}")


if __name__ == "__main__":
    main()
