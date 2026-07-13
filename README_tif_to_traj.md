# TIFF-to-trajectory CLI guide

Run non-DL trajectory extraction for one already-cropped cell. Input consists of
three registered time-lapse TIFF stacks: green, red, and purple.

~~~text
TIFFs -> metadata validation -> trajectory extraction -> review -> final CSVs
~~~

## Scope

Included: preparation of three cropped channels, Stage 1 tracking, MATLAB
Stage 2 2D-Gaussian SPT, Stage 3 matching/filtering, and review figures.

Excluded: ND2 import, cell/nucleus segmentation, Fiji preprocessing, deep
learning, model training, and chromatin-density modelling.

## Requirements

- Python 3.13 (validated version)
- MATLAB available as "matlab" in PowerShell
- A clone of this repository
- Three already-cropped, already-registered TIFF stacks for one cell

MATLAB is external software. It is required for Stage 2.

### Input rules

The three channel TIFFs must have the same cell, identical crop boundaries,
registration/alignment, TYX shape, XY calibration, and frame interval. Do not
crop each channel to a different box: matching uses physical position.

## Quick start

All commands are for PowerShell. Replace every value in angle brackets with
your own path or URL.

### 1. Clone and install

If the repository is already cloned, open PowerShell in its root folder and
start from the "$repo = (Get-Location).Path" line.

~~~powershell
$repositoryUrl = '<repository-url>'                # REPLACE: the GitHub clone URL
$cloneFolder = '<cloned-repository-folder>'        # REPLACE: local folder created/used for this clone
git clone $repositoryUrl $cloneFolder
Set-Location $cloneFolder

# Repository root.
$repo = (Get-Location).Path                        # AUTO: repository root selected above

# New Python environment created inside this clone.
python -m venv .venv
$py = Join-Path $repo '.venv\Scripts\python.exe'  # AUTO: venv executable created above

# Install only the packages in requirements_tif_to_traj.txt.
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $repo 'requirements_tif_to_traj.txt')

# Required on Windows to avoid console-encoding errors in pipeline output.
$env:PYTHONIOENCODING = 'utf-8'

# Preflight checks.
& $py -c "import numpy, scipy, PIL, tifffile, matplotlib; print('Python: OK')"
$matlab = 'matlab'                                 # KEEP if on PATH; otherwise REPLACE with full matlab.exe path
& $matlab -batch "disp('MATLAB: OK')"
~~~

Expected final messages: "Python: OK" and "MATLAB: OK". If MATLAB is not
found, install it or add its executable directory to the system PATH.

### 2. Create a run folder and add input TIFFs

Create one new run folder per cell. The commands create "input/" and
"results/"; the pipeline creates "work/" later.

~~~text
<path-to-run-folder>/
+-- input/       persistent copies of the three TIFFs
+-- work/        temporary pipeline files
+-- results/     final CSVs and review artifacts
~~~

~~~powershell
# New folder for this cell. It can be inside or outside the repository.
$run = '<path-to-run-folder>'                      # REPLACE: new/existing folder dedicated to this one cell
$inputDir = Join-Path $run 'input'                 # AUTO: persistent input copy folder
$resultsDir = Join-Path $run 'results'             # AUTO: accepted output folder
New-Item -ItemType Directory -Path $inputDir, $resultsDir -Force | Out-Null

# Original TIFF locations. Their filenames can be anything.
$greenSource = '<path-to-green-channel-tiff>'      # REPLACE: registered green TYX TIFF
$redSource = '<path-to-red-channel-tiff>'          # REPLACE: registered red TYX TIFF
$purpleSource = '<path-to-purple-channel-tiff>'    # REPLACE: registered purple TYX TIFF

# Copy inputs to standard names. Originals are not modified.
Copy-Item -LiteralPath $greenSource -Destination (Join-Path $inputDir 'green.tif')
Copy-Item -LiteralPath $redSource -Destination (Join-Path $inputDir 'red.tif')
Copy-Item -LiteralPath $purpleSource -Destination (Join-Path $inputDir 'purple.tif')
~~~

If the TIFFs are already in "input/", skip the three copy commands.

### 3. Prepare and validate channels

~~~powershell
$analysis = Join-Path $run 'work\analysis'         # AUTO: temporary analysis directory
$review = Join-Path $resultsDir 'review'           # AUTO: QC output directory
$finalCsv = Join-Path $resultsDir 'cleaned_trajectories' # AUTO: accepted CSV directory

# Validate metadata and create work/analysis.
& $py "$repo\trajectory_extraction\pipeline\prepare_single_channel_analysis.py" --green (Join-Path $inputDir 'green.tif') --red (Join-Path $inputDir 'red.tif') --purple (Join-Path $inputDir 'purple.tif') --output-dir $analysis --stem 'cell'
~~~

Stop if shape, timing, or calibration differs between channels.

Without a real nucleus TIFF, preparation creates a synthetic nucleus scaffold.
It is valid only for this trajectory workflow, never for nuclear or chromatin
measurements. If a real nucleus TIFF has the same crop and registration, add
"--nucleus <path-to-nucleus-tiff>" to the preparation command.

### 4. Run trajectory extraction

~~~powershell
$referenceChannel = 'green'                        # REPLACE if needed: green, red, or purple
$matlabWorkers = 1                                 # KEEP for compatibility; try 2/3 only after benchmarking CPU/RAM
& $py "$repo\trajectory_extraction\run_full_pipeline_v3.py" --no-fiji $analysis `
  --matlab-bin $matlab `
  --matlab-workers $matlabWorkers `
  --reference-channel $referenceChannel
~~~

"--no-fiji" is required because inputs are already separate, registered channel
TIFFs. The command runs Stage 1 tracking, Stage 2 MATLAB SPT, and Stage 3
matching/filtering.

With `--matlab-workers 1`, all three channels reuse one MATLAB session. Values
`2` or `3` launch concurrent MATLAB processes and may reduce wall time, but need
more RAM, CPU, disk bandwidth, and MATLAB license capacity. Full filtered image
movies are no longer saved in MAT files by default because final matching never
reads them; add `--matlab-save-filter-images` only for that diagnostic artifact.

Do not delete "work/" yet. The review command needs its temporary reference and
candidate tracks.

### 5. Review and save final outputs

~~~powershell
# Write QC tables and figures to results/review/.
& $py "$repo\trajectory_extraction\pipeline\review_trajectory_results.py" --analysis-dir $analysis --output-dir $review

# Copy only final Stage-3 trajectory CSVs into results/.
New-Item -ItemType Directory -Path $finalCsv -Force | Out-Null
Copy-Item -LiteralPath (Get-ChildItem -LiteralPath $analysis -Filter '*_traj_m2DGaussian_cleaned.csv' -File | Select-Object -ExpandProperty FullName) -Destination $finalCsv
~~~

Review before accepting:

| File | Check |
| --- | --- |
| "results/review/review_manifest.json" | "status" is "PASS"; calibration agrees across channels. |
| "results/review/match_filter_summary.csv" | Each locus has a saved or rejected reason. |
| "results/review/match_filter_avg_distance.png" | Saved matches are below the 2000 nm distance filter. |
| "results/review/cleaned_tracks_on_channel_mip.png" | Retained tracks plausibly overlay signal in their own channel. |
| "results/cleaned_trajectories/" | Final CSVs with "frame", "x_nm", and "y_nm". |

### 6. Optional cleanup

Keep "work/" while debugging. After review is accepted:

~~~powershell
if (-not (Test-Path -LiteralPath (Join-Path $review 'review_manifest.json'))) { throw 'Review output is missing; do not delete work yet.' }

# Delete only this run's temporary folder. input/ and results/ remain.
Remove-Item -LiteralPath (Join-Path $run 'work') -Recurse -Force
~~~

To rerun the same cell, keep "input/" and "results/", delete "work/", then
start again from Step 3.
