# Changelog

## 4.1.5-experiment-profiles — 2026-07-19

- Fix the legacy MATLAB single-track boundary so every row of the only
  retained trajectory is renumbered consistently as track ID 1.
- Extend the Sherlock MATLAB regression to cover zero retained trajectories
  and exactly one retained trajectory without changing scientific parameters.

## 4.1.4-experiment-profiles — 2026-07-18

- Assign the correctly shaped empty matrix to the declared MATLAB `tracks`
  output before the legacy `track.m` empty-result branch returns.
- Extend the regression contract to prevent an unassigned-output failure while
  preserving all v4.1.3 tracking parameters and locked experiment profiles.

## 4.1.3-experiment-profiles — 2026-07-18

- Return an empty matrix from legacy MATLAB `track.m` before its post-processing
  indexes the first row when no tracks survive linking or length filtering.
- Keep the v4.1.2 common `spt_track.m` empty-result handling and validate both
  layers with the Sherlock MATLAB regression smoke test.

## 4.1.2-experiment-profiles — 2026-07-18

- Treat an empty MATLAB `track.m` result as a valid zero-trajectory outcome
  for both consecutive- and non-consecutive-frame inputs.
- Preserve empty channel/allele results without changing thresholds, minimum
  trajectory length or other scientific tracking parameters.
- Add a MATLAB regression smoke test for multi-frame detections that produce
  no retained trajectories.

## 4.1.1-experiment-profiles — 2026-07-18

- Preserved the two locked v4.1 biological experiment profiles while adding
  three runtime fixes recovered from the audited Sherlock smoke-test clone.
- Return an explicit empty trajectory result when legacy MATLAB SPT receives
  detections from fewer than two frames.
- Normalize literal `\\u00b5`/`\\u03bc` TIFF spatial-unit metadata before the
  physical max-step validation.
- Read baseline CSV manifests with an explicit text/CSV contract and export
  SVG through the MATLAB R2022b-compatible `print -dsvg` path.

## 4.1.0-experiment-profiles — 2026-07-18

- Added two and only two locked biological acquisition profiles:
  `chr3_sites_2_3_4` and `dsb_53bp1_site1_site2`.
- Locked raw C2 as the anchor in both profiles while preserving its distinct
  biological identity: Chr3 Site 2/A488 for the four-channel manuscript data,
  and Site 2/Purple for the three-channel DSB data.
- Added hard profile validation for channel count and filename evidence before
  Fiji or MATLAB starts.
- Required the Step-3 acquisition sidecar and validated the original ND2
  channel-name order (`405/640/488/561` or `GFP/RFP/Cy5`) before execution.
- Removed the free-form production anchor override and moved marker names,
  slugs, fluorophores, genomic loci and raw indices into the profile contract.
- Namespaced outputs by profile so results from the two experiments cannot
  overwrite or be resumed as each other.
- Removed hard-coded 53BP1/Site1/Site2 labels from CSV, Python QC and MATLAB
  figure generation.

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
