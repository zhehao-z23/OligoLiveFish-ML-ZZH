"""
Posthoc QC utilities for fixed LiveFISH cell-crop outputs.

This module applies the locked low-quality nucleus review rule:

    bad_qc = (ch0_contrast < 0.085) & (ch0_boundary_grad < 1.70)

It expects metrics CSV files that already contain `cell_id`, `ch0_contrast`,
and `ch0_boundary_grad`. The segmentation/cropping script remains responsible
for producing candidate crops; this QC layer marks crops whose channel-0 nuclear
signal is both low contrast and weak at the mask boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


BAD_QC_CH0_CONTRAST_MAX = 0.085
BAD_QC_CH0_BOUNDARY_GRAD_MAX = 1.70


def add_bad_qc_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the locked `bad_qc` rule applied."""
    required = {"ch0_contrast", "ch0_boundary_grad"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metric column(s): {', '.join(missing)}")

    out = df.copy()
    out["bad_qc"] = (
        (out["ch0_contrast"].astype(float) < BAD_QC_CH0_CONTRAST_MAX)
        & (out["ch0_boundary_grad"].astype(float) < BAD_QC_CH0_BOUNDARY_GRAD_MAX)
    )
    out["bad_qc_rule"] = (
        f"(ch0_contrast < {BAD_QC_CH0_CONTRAST_MAX}) & "
        f"(ch0_boundary_grad < {BAD_QC_CH0_BOUNDARY_GRAD_MAX})"
    )
    return out


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _plot_rule_boundary(ax):
    ax.axvline(
        BAD_QC_CH0_CONTRAST_MAX,
        color="#8b1a1a",
        linestyle="--",
        linewidth=1.2,
        label=f"ch0_contrast < {BAD_QC_CH0_CONTRAST_MAX}",
    )
    ax.axhline(
        BAD_QC_CH0_BOUNDARY_GRAD_MAX,
        color="#8b1a1a",
        linestyle=":",
        linewidth=1.2,
        label=f"ch0_boundary_grad < {BAD_QC_CH0_BOUNDARY_GRAD_MAX}",
    )
    ax.add_patch(
        Rectangle(
            (0, 0),
            BAD_QC_CH0_CONTRAST_MAX,
            BAD_QC_CH0_BOUNDARY_GRAD_MAX,
            facecolor="#f8d7da",
            edgecolor="none",
            alpha=0.28,
            zorder=0,
        )
    )
    ax.text(
        BAD_QC_CH0_CONTRAST_MAX * 0.48,
        BAD_QC_CH0_BOUNDARY_GRAD_MAX * 0.45,
        "bad_qc\nregion",
        ha="center",
        va="center",
        fontsize=8,
        color="#8b1a1a",
    )


def plot_bad_qc_scatter(
    fov17: pd.DataFrame,
    out_path: Path,
    fov7: pd.DataFrame | None = None,
    title: str = "Cell-crop ch0 QC rule",
) -> None:
    """Write a scatter plot with the locked bad_qc threshold boundary."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _plot_rule_boundary(ax)

    problem = (
        _as_bool_series(fov17["is_problem_cell"])
        if "is_problem_cell" in fov17.columns
        else pd.Series(False, index=fov17.index)
    )
    bad = _as_bool_series(fov17["bad_qc"])
    normal = ~problem & ~bad

    if normal.any():
        ax.scatter(
            fov17.loc[normal, "ch0_contrast"],
            fov17.loc[normal, "ch0_boundary_grad"],
            s=44,
            c="#4c78a8",
            edgecolors="white",
            linewidths=0.5,
            label="fov17 final crop",
            alpha=0.9,
        )
    if problem.any():
        ax.scatter(
            fov17.loc[problem, "ch0_contrast"],
            fov17.loc[problem, "ch0_boundary_grad"],
            s=58,
            c="#f58518",
            edgecolors="#4a2a00",
            linewidths=0.7,
            label="fov17 review cell",
            alpha=0.95,
        )
    if bad.any():
        ax.scatter(
            fov17.loc[bad, "ch0_contrast"],
            fov17.loc[bad, "ch0_boundary_grad"],
            s=88,
            facecolors="none",
            edgecolors="#b00020",
            linewidths=1.8,
            label="bad_qc hit",
        )

    if fov7 is not None and not fov7.empty:
        fov7_bad = _as_bool_series(fov7["bad_qc"]) if "bad_qc" in fov7.columns else pd.Series(False, index=fov7.index)
        ax.scatter(
            fov7.loc[~fov7_bad, "ch0_contrast"],
            fov7.loc[~fov7_bad, "ch0_boundary_grad"],
            s=42,
            c="#54a24b",
            marker="^",
            edgecolors="white",
            linewidths=0.5,
            label="fov7 accepted validation",
            alpha=0.85,
        )
        if fov7_bad.any():
            ax.scatter(
                fov7.loc[fov7_bad, "ch0_contrast"],
                fov7.loc[fov7_bad, "ch0_boundary_grad"],
                s=92,
                facecolors="none",
                edgecolors="#b00020",
                marker="^",
                linewidths=1.8,
                label="fov7 bad_qc hit",
            )

    for _, row in fov17.loc[bad].iterrows():
        ax.annotate(
            str(row.get("cell_id", "")),
            (float(row["ch0_contrast"]), float(row["ch0_boundary_grad"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#5c000b",
        )

    all_x = [fov17["ch0_contrast"].astype(float)]
    all_y = [fov17["ch0_boundary_grad"].astype(float)]
    if fov7 is not None and not fov7.empty:
        all_x.append(fov7["ch0_contrast"].astype(float))
        all_y.append(fov7["ch0_boundary_grad"].astype(float))
    xmax = max(float(pd.concat(all_x).max()) * 1.08, BAD_QC_CH0_CONTRAST_MAX * 2)
    ymax = max(float(pd.concat(all_y).max()) * 1.08, BAD_QC_CH0_BOUNDARY_GRAD_MAX * 1.6)

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_xlabel("ch0_contrast")
    ax.set_ylabel("ch0_boundary_grad")
    ax.set_title(title)
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(out_dir: Path, fov17: pd.DataFrame, fov7: pd.DataFrame | None) -> None:
    fov17_hits = fov17.loc[_as_bool_series(fov17["bad_qc"]), "cell_id"].astype(str).tolist()
    fov7_hits: list[str] = []
    if fov7 is not None and not fov7.empty:
        fov7_hits = fov7.loc[_as_bool_series(fov7["bad_qc"]), "cell_id"].astype(str).tolist()

    lines = [
        "# Cell-Crop Bad QC Rule",
        "",
        "Locked rule:",
        "",
        "```python",
        f"bad_qc = (ch0_contrast < {BAD_QC_CH0_CONTRAST_MAX}) & "
        f"(ch0_boundary_grad < {BAD_QC_CH0_BOUNDARY_GRAD_MAX})",
        "```",
        "",
        f"FOV17 bad_qc hits: {', '.join(fov17_hits) if fov17_hits else 'none'}.",
    ]
    if fov7 is not None:
        lines.append(f"FOV7 validation hits: {', '.join(fov7_hits) if fov7_hits else 'none'}.")
    lines.extend(
        [
            "",
            "The rule is intended for final-crop nuclear quality review, not for",
            "micro-SAM instance generation. Structural segmentation filters remain",
            "in `crop_nuclei_sam.py`.",
            "",
        ]
    )
    (out_dir / "bad_qc_rule_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply and visualize the locked cell-crop bad_qc rule.")
    parser.add_argument("--fov17-metrics", required=True, help="CSV with fov17 ch0_contrast/ch0_boundary_grad metrics")
    parser.add_argument("--fov7-metrics", default="", help="Optional CSV with fov7 validation metrics")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV and PNG artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fov17 = add_bad_qc_rule(pd.read_csv(args.fov17_metrics))
    fov17.to_csv(out_dir / "fov17_bad_qc_rule_results.csv", index=False)

    fov7 = None
    if args.fov7_metrics:
        fov7 = add_bad_qc_rule(pd.read_csv(args.fov7_metrics))
        fov7.to_csv(out_dir / "fov7_bad_qc_rule_validation.csv", index=False)

    plot_bad_qc_scatter(
        fov17,
        out_dir / "bad_qc_scatter_fov17_only.png",
        title="FOV17 ch0 QC rule boundary",
    )
    if fov7 is not None:
        plot_bad_qc_scatter(
            fov17,
            out_dir / "bad_qc_scatter_fov17_with_fov7_validation.png",
            fov7=fov7,
            title="FOV17 rule with FOV7 accepted-crop validation",
        )
    write_summary(out_dir, fov17, fov7)
    print(f"Cell-crop bad_qc outputs written to {out_dir}")


if __name__ == "__main__":
    main()
