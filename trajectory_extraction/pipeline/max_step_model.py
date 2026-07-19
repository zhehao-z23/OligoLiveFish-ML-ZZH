#!/usr/bin/env python3
"""Metadata-to-physics model for the v4 trajectory-linking step radius."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import tifffile


MODEL_VERSION = "v4.0.0-anchor-roi"


def _resolution_value(value) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def read_tracking_metadata(tiff_path: Path) -> dict:
    """Read calibrated time and isotropic x/y pixel size from one channel TIFF."""
    tiff_path = Path(tiff_path).resolve()
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]
        imagej = tif.imagej_metadata or {}
        description = page.description or ""

        frame_interval_s = imagej.get("finterval")
        if frame_interval_s is None:
            match = re.search(r"finterval=([0-9.eE+\-]+)", description)
            if not match:
                raise ValueError(f"finterval not found in TIFF metadata: {tiff_path}")
            frame_interval_s = float(match.group(1))
        frame_interval_s = float(frame_interval_s)
        if frame_interval_s <= 0:
            raise ValueError(f"finterval must be positive: {frame_interval_s}")

        if "XResolution" not in page.tags:
            raise ValueError(f"XResolution not found in TIFF metadata: {tiff_path}")
        x_resolution_px_per_um = _resolution_value(page.tags["XResolution"].value)
        y_resolution_px_per_um = (
            _resolution_value(page.tags["YResolution"].value)
            if "YResolution" in page.tags
            else x_resolution_px_per_um
        )
        pixel_size_x_um = 1.0 / x_resolution_px_per_um
        pixel_size_y_um = 1.0 / y_resolution_px_per_um
        if not math.isclose(pixel_size_x_um, pixel_size_y_um, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                "The scalar radial linker requires isotropic pixels; "
                f"found x={pixel_size_x_um} and y={pixel_size_y_um} um/px"
            )

        unit = str(imagej.get("unit", ""))
        unit = re.sub(r"\\u(?:00b5|03bc)", "u", unit, flags=re.IGNORECASE)
        if unit.lower() not in {"micron", "microns", "um", "µm", "μm"}:
            raise ValueError(
                "TIFF spatial unit must explicitly be microns; "
                f"found {unit!r} in {tiff_path}"
            )

        series = tif.series[0]
        axes = series.axes
        shape = tuple(int(value) for value in series.shape)
        frame_count = shape[axes.index("T")] if "T" in axes else len(tif.pages)
        return {
            "source_tiff": str(tiff_path),
            "frame_interval_s": frame_interval_s,
            "frame_rate_hz": 1.0 / frame_interval_s,
            "x_resolution_px_per_um": x_resolution_px_per_um,
            "y_resolution_px_per_um": y_resolution_px_per_um,
            "pixel_size_x_um_per_px": pixel_size_x_um,
            "pixel_size_y_um_per_px": pixel_size_y_um,
            "pixel_size_nm_per_px": pixel_size_x_um * 1000.0,
            "spatial_unit": unit,
            "frame_count": int(frame_count),
            "series_axes": axes,
            "series_shape": shape,
            "isotropic_pixels_verified": True,
        }


def validate_channel_metadata(channel_tiffs: dict[str, Path]) -> dict[str, dict]:
    """Require G/R/P TIFFs to share the calibration used by one scalar model."""
    metadata = {
        channel: read_tracking_metadata(path) for channel, path in channel_tiffs.items()
    }
    if not metadata:
        raise ValueError("At least one channel TIFF is required")
    canonical_channel = next(iter(metadata))
    canonical = metadata[canonical_channel]
    numeric_fields = (
        "frame_interval_s",
        "pixel_size_x_um_per_px",
        "pixel_size_y_um_per_px",
    )
    exact_fields = ("spatial_unit", "frame_count")
    for channel, item in metadata.items():
        for field in numeric_fields:
            if not math.isclose(item[field], canonical[field], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"{channel} TIFF {field} differs from {canonical_channel}: "
                    f"{item[field]} != {canonical[field]}"
                )
        for field in exact_fields:
            if item[field] != canonical[field]:
                raise ValueError(
                    f"{channel} TIFF {field} differs from {canonical_channel}: "
                    f"{item[field]!r} != {canonical[field]!r}"
                )
    return metadata


def calculate_max_displacement(
    *,
    pixel_size_um: float,
    frame_interval_s: float,
    frame_gap: int,
    diffusion_coefficient_um2_per_s_alpha: float,
    anomalous_exponent: float,
    coverage_probability: float,
    localization_error_um: float,
) -> dict:
    """Calculate a Rayleigh radial quantile for 2-D anomalous diffusion."""
    if pixel_size_um <= 0 or frame_interval_s <= 0:
        raise ValueError("pixel size and frame interval must be positive")
    if frame_gap < 1:
        raise ValueError("frame_gap must be at least 1")
    if diffusion_coefficient_um2_per_s_alpha < 0 or localization_error_um < 0:
        raise ValueError("diffusion coefficient and localization error cannot be negative")
    if not 0 < anomalous_exponent <= 2:
        raise ValueError("anomalous_exponent must be in (0, 2]")
    if not 0 < coverage_probability < 1:
        raise ValueError("coverage_probability must be in (0, 1)")

    lag_time_s = frame_gap * frame_interval_s
    diffusion_term_um2 = (
        diffusion_coefficient_um2_per_s_alpha * lag_time_s**anomalous_exponent
    )
    localization_term_um2 = localization_error_um**2
    radial_variance_term_um2 = diffusion_term_um2 + localization_term_um2
    if radial_variance_term_um2 <= 0:
        raise ValueError(
            "At least one modeled motion/localization variance term must be positive"
        )
    radius_um = math.sqrt(
        -4.0 * math.log(1.0 - coverage_probability) * radial_variance_term_um2
    )
    return {
        "frame_gap": frame_gap,
        "lag_time_s": lag_time_s,
        "diffusion_term_um2": diffusion_term_um2,
        "localization_term_um2": localization_term_um2,
        "radial_variance_term_um2": radial_variance_term_um2,
        "theoretical_radius_um": radius_um,
        "theoretical_radius_nm": radius_um * 1000.0,
        "theoretical_radius_px": radius_um / pixel_size_um,
    }


def ceil_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        raise ValueError("rounding increment must be positive")
    return math.ceil((value - 1e-12) / increment) * increment


def derive_from_metadata(
    metadata: dict,
    *,
    diffusion_coefficient_um2_per_s_alpha: float = 4.1e-3,
    anomalous_exponent: float = 0.38,
    coverage_probability: float = 0.995,
    localization_error_nm: float = 0.0,
    frame_gap: int = 1,
    rounding_increment_px: float = 0.05,
    track_mem: int = 3,
    explicit_max_step_px: float | None = None,
) -> dict:
    """Combine TIFF metadata and declared priors into an operational max step."""
    theoretical = calculate_max_displacement(
        pixel_size_um=metadata["pixel_size_x_um_per_px"],
        frame_interval_s=metadata["frame_interval_s"],
        frame_gap=frame_gap,
        diffusion_coefficient_um2_per_s_alpha=diffusion_coefficient_um2_per_s_alpha,
        anomalous_exponent=anomalous_exponent,
        coverage_probability=coverage_probability,
        localization_error_um=localization_error_nm / 1000.0,
    )
    modeled_px = ceil_to_increment(
        theoretical["theoretical_radius_px"], rounding_increment_px
    )
    if explicit_max_step_px is not None:
        if explicit_max_step_px <= 0:
            raise ValueError("explicit_max_step_px must be positive")
        operational_px = float(explicit_max_step_px)
        operational_source = "explicit CLI override"
    else:
        operational_px = modeled_px
        operational_source = "metadata + physical prior + upward rounding"

    operational_um = operational_px * metadata["pixel_size_x_um_per_px"]
    variance_term = theoretical["radial_variance_term_um2"]
    achieved_coverage = 1.0 - math.exp(-(operational_um**2) / (4.0 * variance_term))
    gap_sensitivity = [
        calculate_max_displacement(
            pixel_size_um=metadata["pixel_size_x_um_per_px"],
            frame_interval_s=metadata["frame_interval_s"],
            frame_gap=gap,
            diffusion_coefficient_um2_per_s_alpha=diffusion_coefficient_um2_per_s_alpha,
            anomalous_exponent=anomalous_exponent,
            coverage_probability=coverage_probability,
            localization_error_um=localization_error_nm / 1000.0,
        )
        for gap in range(1, track_mem + 2)
    ]
    return {
        "version": MODEL_VERSION,
        "formula": "r_p(tau)=sqrt(-4*ln(1-p)*(D_star*tau^alpha+sigma_loc^2)); max_step_px=r_p/pixel_size_um",
        "metadata": metadata,
        "physical_prior": {
            "diffusion_coefficient_D_star_um2_per_s_alpha": diffusion_coefficient_um2_per_s_alpha,
            "anomalous_exponent_alpha": anomalous_exponent,
            "radial_coverage_probability_p": coverage_probability,
            "localization_error_per_frame_per_axis_nm": localization_error_nm,
        },
        "applied_lag": {"frame_gap": frame_gap, "lag_time_s": theoretical["lag_time_s"]},
        "calculation": theoretical,
        "rounding_policy": {"mode": "ceiling", "increment_px": rounding_increment_px},
        "modeled_max_step_px": modeled_px,
        "explicit_max_step_px": explicit_max_step_px,
        "operational_source": operational_source,
        "operational_max_step_px": operational_px,
        "operational_max_step_um": operational_um,
        "operational_max_step_nm": operational_um * 1000.0,
        "achieved_adjacent_frame_radial_coverage": achieved_coverage,
        "tracker_implementation": {
            "track_mem": track_mem,
            "maximum_detection_frame_gap": track_mem + 1,
            "gap_scaled_radius_implemented": False,
            "warning": "the same scalar radius is used after gaps; sensitivity values are audit-only",
        },
        "gap_sensitivity_not_applied": gap_sensitivity,
    }


def derive_from_tiff(tiff_path: Path, **kwargs) -> dict:
    return derive_from_metadata(read_tracking_metadata(tiff_path), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--d-star", type=float, default=4.1e-3)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--coverage", type=float, default=0.995)
    parser.add_argument("--localization-error-nm", type=float, default=0.0)
    parser.add_argument("--frame-gap", type=int, default=1)
    parser.add_argument("--rounding-increment-px", type=float, default=0.05)
    parser.add_argument("--track-mem", type=int, default=3)
    parser.add_argument("--max-step-px", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = derive_from_tiff(
        args.tiff,
        diffusion_coefficient_um2_per_s_alpha=args.d_star,
        anomalous_exponent=args.alpha,
        coverage_probability=args.coverage,
        localization_error_nm=args.localization_error_nm,
        frame_gap=args.frame_gap,
        rounding_increment_px=args.rounding_increment_px,
        track_mem=args.track_mem,
        explicit_max_step_px=args.max_step_px,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
