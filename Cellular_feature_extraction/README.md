# Feature Extraction

Extracts spatial and morphological features from nuclear segmentation masks and the location of DNA locus.

## Scripts

**`extract_features.py`** -- Core feature extractor. For each nucleus folder, computes:
- Nuclear morphology: area, perimeter, circularity, eccentricity, solidity, axis lengths
- Locus spatial context: distance to nuclear membrane, distance to centroid, normalized radial position (0=center, 1=periphery)
- Local chromatin environment: mean DAPI intensity in a window around each locus, normalized by nuclear mean

```bash
python3 extract_features.py /path/to/nucleus_folder [--pixel-size 108.33]
```

**`extract_nuclear_features.py`** -- Standalone nuclear morphology extraction from binary masks (`<stem>_mask_<num>.tif`). Outputs per-frame and per-nucleus summary features.

```bash
python3 extract_nuclear_features.py --data-root /path/to/data
```

## Output

- `locus_features.csv` -- per-locus, per-frame spatial features
- `nucleus_features.csv` -- per-frame nuclear morphology and intensity

Trajectory coordinates are registered to segmentation masks to verify that tracked loci fall within nuclear boundaries (83.9% of observations pass this check).
