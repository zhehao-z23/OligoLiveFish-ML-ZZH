# Nucleus-cropping documentation moved

The production nucleus-cropping scripts live in `nucleus_segmentation/`; this
pipeline directory retains compatibility copies for historical runs only.

Use these maintained documents:

- [`../../README.md`](../../README.md) for the complete ND2-to-cleaned-
  trajectory tutorial, runnable PowerShell commands, parameters, GPU selection,
  QC, batching, and troubleshooting;
- [`../../REQUIREMENTS_ND2_TO_TRAJ.md`](../../REQUIREMENTS_ND2_TO_TRAJ.md)
  for the venv and optional CUDA setup;
- [`../../nucleus_segmentation/crop_nucleus.md`](../../nucleus_segmentation/crop_nucleus.md)
  for the focused cross-platform nucleus-crop notes.

Do not copy old developer-specific paths or edit the Python source to select a
device. Use the `--device auto|cuda|mps|cpu` CLI option.
