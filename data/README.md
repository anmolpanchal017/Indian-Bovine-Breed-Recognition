# Data Directory

## Directory Contract

- `raw/images/`: untouched copies of source images. Never overwrite raw files during preprocessing.
- `external/`: data from outside this project, with source and license notes.
- `metadata/`: normalized CSV/JSON metadata, schemas, and data dictionaries.
- `processed/`: derived images or tensors generated from raw data.
- `splits/`: train, validation, and test manifests. Store paths and labels, not duplicated images, unless a workflow requires materialized subsets.

## Current-to-Canonical Mapping

The current dataset has not been moved automatically:

```text
Current source images: Indian_bovine_breeds/Indian_bovine_breeds/<breed>/...
Current metadata:      bovine_breeds_metadata.csv
Canonical raw target:  data/raw/images/<breed>/...
Canonical metadata:    data/metadata/bovine_breeds_metadata.csv
```

Until migration is completed, tools must use the current source locations. Do not assume that `data/raw/images/` already contains the images.

## Path Rules

Metadata paths must be relative to the dataset root configured by the consuming script. Prefer `/` separators in committed metadata. Avoid absolute paths because they only work on one machine.
