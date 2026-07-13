# Environment Guide: ND2 to Cleaned Trajectories

This document describes only the runtime environment for the preprocessing
workflow documented in [README.md](README.md). The machine-readable Python
dependencies are in
[`requirements_nd2_to_traj.txt`](requirements_nd2_to_traj.txt).

## What is and is not installed by pip

The requirements file installs the Python libraries used for ND2 reading,
micro-SAM nucleus segmentation, TIFF processing, reference tracking, MATLAB
result loading, and trajectory QC.

It does not install these external applications:

- MATLAB, required for 2D-Gaussian SPT;
- Fiji/ImageJ with **Correct 3D drift**, required when starting from a supported
  three- or four-channel crop TIFF produced from ND2.

PyTorch is included because it is the execution backend for micro-SAM. Its
presence does not run or install this repository's downstream deep-learning
experiments.

## Validated environment

The pinned direct dependencies were validated in a fresh Windows venv with
`pip check` and the project's CLI import paths:

```text
Python 3.13.14
numpy 2.4.6
scipy 1.18.0
Pillow 12.2.0
matplotlib 3.11.0
scikit-image 0.26.0
tifffile 2026.6.1
nd2 0.11.3
dask 2026.6.0
torch 2.10.0
micro-SAM 1.8.4
```

Use Python 3.13 for the documented reproducible setup. A newer or older Python
may require a different dependency set and should be validated as a separate
environment.

`nd2==0.11.1`, used by an earlier working environment, is not retained here:
PyPI withdrew that release for an experiment-loop detection regression. The
fresh environment uses the current non-yanked `nd2==0.11.3`; see the
[official PyPI release history](https://pypi.org/project/nd2/0.11.3/).

## Windows PowerShell setup

Run these commands from the repository root. The environment is stored inside
the repository as `.venv_nd2_to_traj`; that folder is local runtime state and
must not be committed. The commands do not require the optional Windows `py`
launcher and do not require environment activation.

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$Repo = (Get-Location).Path                              # AUTO: first Set-Location to this repository root
$BasePython = (Get-Command python -ErrorAction Stop).Source # AUTO: selected base python.exe
& $BasePython --version
& $BasePython -m venv "$Repo\.venv_nd2_to_traj"
if ($LASTEXITCODE -ne 0) { throw "venv creation failed; do not install packages" }

$Python = "$Repo\.venv_nd2_to_traj\Scripts\python.exe" # AUTO: venv executable created above
if (-not (Test-Path -LiteralPath $Python)) { throw "venv Python was not created: $Python" }
& $Python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix, 'Not running inside a venv'"
if ($LASTEXITCODE -ne 0) { throw "venv isolation check failed" }

& $Python -m pip install --upgrade pip
& $Python -m pip install -r "$Repo\requirements_nd2_to_traj.txt"
```

The UTF-8 settings prevent non-ASCII scientific units in console output from
causing `UnicodeEncodeError` or mojibake on legacy Windows code pages.

In each later PowerShell session, point `$Python` to the existing venv. All
project commands can use this path directly, so activation is optional:

```powershell
$Repo = (Get-Location).Path                              # AUTO: current repository root
$Python = "$Repo\.venv_nd2_to_traj\Scripts\python.exe" # AUTO: reuse, do not recreate, this venv
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $Python --version
```

If activation is desired for interactive use, first verify that the venv exists,
then run `Set-ExecutionPolicy -Scope Process Bypass` followed by
`& "$Repo\.venv_nd2_to_traj\Scripts\Activate.ps1"`.

### Why this guide does not start with `py -3.13`

`py.exe` is the separate Python Launcher for Windows. Microsoft Store Python
can provide a working `python.exe` without providing `py.exe`. If `py` is not
recognized, use `python -m venv` as above; do not continue to the install step
until `.venv_nd2_to_traj\Scripts\python.exe` exists.

If pip prints `Defaulting to user installation`, stop: the command is not using
the venv. Always install with `& $Python -m pip ...`. If an accidental user-site
installation already completed, leave it in place unless its packages are
known to be unused; the new venv is isolated from user-site packages.

## macOS or Linux setup

```bash
python3.13 -m venv .venv_nd2_to_traj
source .venv_nd2_to_traj/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_nd2_to_traj.txt
```

Activate it in later shells with:

```bash
source .venv_nd2_to_traj/bin/activate
```

## Verify the Python environment

```powershell
& $Python --version
& $Python -c "import sys, nd2, micro_sam, numpy, scipy, skimage, tifffile, torch; assert sys.prefix != sys.base_prefix; print('Python imports: OK'); print('nd2:', nd2.__version__); print('CUDA available:', torch.cuda.is_available()); print(sys.executable)"
& $Python -m pip --version
& $Python -m pip check
```

`CUDA available: False` is not an installation failure: segmentation can run on
CPU with `--device cpu` or `--device auto`, although it is slower. GPU execution
requires a PyTorch build compatible with the computer's GPU driver. Treat any
GPU-specific installation as a separately recorded environment and rerun the
three verification commands afterward.

### Optional NVIDIA GPU backend for large segmentation batches

`requirements_nd2_to_traj.txt` is the reproducible baseline and remains usable
on CPU. A CUDA wheel cannot be selected universally in that file because the
correct build depends on the operating system, NVIDIA driver, and supported
CUDA version.

For a dedicated GPU environment:

1. Finish the normal venv installation above and confirm the CPU workflow can
   import all packages.
2. Open the official [PyTorch installation
   selector](https://pytorch.org/get-started/locally/), select the current OS,
   `Pip`, Python, and the CUDA version supported by the installed driver.
3. Remove only the venv's baseline torch wheel with
   `& $Python -m pip uninstall -y torch`; this does not touch global Python.
4. Copy the selector's generated install command, but run it through this venv's
   interpreter. In other words, replace its leading `pip` with
   `& $Python -m pip`. Do not paste a CUDA index URL from another computer.
5. Run the checks below and require `CUDA available: True` before selecting
   `--device cuda`.

```powershell
nvidia-smi
& $Python -c "import torch; print('torch:', torch.__version__); print('torch CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
& $Python -m pip check
```

Use `--device auto` when the same command must work on mixed CPU/GPU machines;
it selects CUDA, then Apple MPS, then CPU. Use explicit `--device cuda` for a
large validated NVIDIA batch so a misconfigured GPU environment fails early
instead of silently running slowly on CPU. MATLAB Stage 2 does not use this
PyTorch/CUDA backend.

## Verify MATLAB and Fiji

MATLAB must resolve from the same terminal used to run Python:

```powershell
$Matlab = "matlab"                                      # KEEP if on PATH; otherwise REPLACE with full matlab.exe path
& $Matlab -batch "disp(['MATLAB ', version])"
```

On Windows, Fiji may be passed by absolute path rather than added to `PATH`:

```powershell
$Fiji = "C:\path\to\Fiji.app\ImageJ-win64.exe"         # REPLACE: Fiji executable, not the Fiji.app directory
Test-Path $Fiji
```

Open Fiji in graphical mode once and confirm that **Correct 3D drift** is
available before launching a headless analysis.

## Record an environment snapshot

The requirements file records the validated direct packages. For each published
or archived run, also capture the complete resolved environment:

```powershell
& $Python -m pip freeze | Out-File -Encoding utf8 environment_snapshot.txt
& $Python --version | Out-File -Encoding utf8 python_version.txt
```

Store those snapshot files with the run metadata, not by overwriting
`requirements_nd2_to_traj.txt`.

## Update policy

Do not upgrade packages inside an environment used for an active dataset. To
test new package versions:

1. create a second venv with a different name;
2. install and record the proposed versions there;
3. rerun a known representative ND2 and compare segmentation, calibration,
   reference tracks, MATLAB candidates, and cleaned outputs;
4. update the pinned requirements only after the comparison is accepted.

## Remove the venv

The venv contains no source data or analysis results. Deactivate it before
removal:

```powershell
deactivate
```

Then delete only the repository's `.venv_nd2_to_traj` directory using the
operating system's file manager. Recreate it later from the pinned requirements.
