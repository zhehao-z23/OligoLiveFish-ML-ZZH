# Trajectory extraction — production v4.1 experiment profiles

Production entry points:

- `run_full_pipeline_v4.py`: one Step-3 crop, optional Fiji preprocessing,
  micro-SAM alignment, profile-locked anchor, static irregular ROI SPT, longest cleaned
  baseline, and QC.
- `run_batch_pipeline_v4.py`: validated multi-cell wrapper with dry-run, resume,
  bounded cell/MATLAB concurrency, and a batch summary.

Implementation modules and bundled MATLAB dependencies live in `pipeline/`.
The authoritative usage guide, all parameters, output tree, and scientific
rules are in [`../README.md`](../README.md).

## Algorithm boundary

```text
saved micro-SAM instance mask
  -> per-frame drift alignment and 5 px dilation
  -> profile validation and locked-anchor filtering
  -> complete anchor path -> connected 5 px static ROI
  -> ROI-gated peak admission -> full-window 2-D Gaussian fit
  -> metadata/physics-derived max-step linking
  -> export every candidate
  -> deterministic longest candidate per allele/channel
```

No manually cleaned file participates in detection, ROI construction,
candidate filtering, or baseline selection. The old v3 reference-distance
matcher is not called by either v4 entry point.

For a manuscript comparison, keep the three data layers distinct:

1. published trajectory files (external reference data);
2. the repository's historical v2.13/v3 published-style implementation;
3. the production v4.1 profile-locked implementation.

Do not write layers 2 and 3 into the same result directory or describe the
historical implementation as byte-identical reproduction without validating it
against the released trajectories.

## Direct module interfaces

These are mainly useful for development and controlled reruns:

| Module | Role |
| --- | --- |
| `experiment_profiles.py` | The only two accepted biological channel contracts and their locked anchors. |
| `align_microsam_mask.py` | Associate and drift-align the saved instance mask. |
| `auto_roi_for_published_v2.13.py --nucleus-mask ...` | Detect/filter the selected anchor using the supplied mask. |
| `max_step_model.py` | Audit metadata and physical-prior conversion to pixels. |
| `run_anchor_roi_spt.py` | Create static ROIs, run MATLAB SPT, audit candidates, select baselines. |
| `visualize_anchor_roi_results.py` | Read-only Python spatial QC. |
| `plot_longest_trajectories.m` | Time-coloured PNG/SVG for longest baselines. |

Direct module use must preserve the same inputs and audit outputs as the
production runner. Dataset-specific experiments belong outside this repository.
