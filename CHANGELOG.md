# Changelog

## 4.0.0-anchor-roi — 2026-07-14

This release promotes the validated FOV15 static anchor-ROI experiment to the
production ND2-to-trajectory workflow.

### Changed

- Fixed the nucleus-mask handoff: every exported crop records its exact
  micro-SAM instance mask, which is aligned to Fiji-corrected frames and passed
  explicitly to anchor tracking. The production path no longer silently
  replaces this boundary with intensity/Otsu segmentation.
- Made Purple the default anchor channel. Accepted complete anchor paths define
  one connected, 5 px-dilated static irregular ROI, intersected with the aligned
  micro-SAM support and reused for every movie frame.
- Restricted SPT peak admission to the anchor ROI while retaining the unmasked
  image pixels required by the 2-D Gaussian fitting window.
- Replaced v3 reference-distance matching, the 2,000 nm filter, and greedy
  assignment in the production path with a deterministic per-allele/channel
  longest-candidate baseline. All candidates remain available for audit.
- Exposed the linking step radius as a metadata-to-physics model with declared
  `D*`, anomalous exponent, coverage probability, localization error, lag, and
  rounding parameters; an explicit CLI override remains available and audited.
- Added automatic Python spatial QC and MATLAB time-coloured PNG/SVG plots of
  selected baselines. SVG trajectories use explicit RGB line segments.
- Added new single-cell and multi-cell production entry points:
  `run_full_pipeline_v4.py` and `run_batch_pipeline_v4.py`.

### Validated regression

- FOV15 metadata produces a theoretical `2.726893 px` radius and the approved
  upward-rounded operational value `2.75 px`.
- micro-SAM alignment is pixel-identical to the approved experimental aligned
  mask.
- Purple anchor outputs for loci 2, 3, and 5 are byte-identical to Run 03.
- All 27 ROI-SPT candidate CSVs are byte-identical to Run 03.
- The deterministic baseline selects 6 available allele/channel tracks and
  records the 3 no-candidate combinations.
- MATLAB export produces 6 PNG and 6 coloured SVG figures for those baselines.

### Migration

The former v3 top-level runners are removed from the production interface.
Low-level v3 matching modules remain in `trajectory_extraction/pipeline/` only
for historical reproducibility; the v4 runners never call them. Existing v3
result folders are not upgraded in place—archive them and rerun from the
Step-3 crop TIFF and associated micro-SAM mask.
