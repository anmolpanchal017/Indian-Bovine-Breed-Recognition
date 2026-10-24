# Project Architecture

## Purpose

This project classifies images of Indian and international bovine breeds. The repository separates source data, derived data, code, experiments, model outputs, and reports so that a person or AI assistant can identify what is safe to modify and what must be regenerated.

## Repository Tree

```text
project/
├── README.md                         # Human entry point
├── PROJECT_ARCHITECTURE.md           # This complete architecture and workflow contract
├── DATASET_CARD.md                   # Dataset scope, limitations, and intended use
├── audit_dataset.py                  # Metadata and image integrity audit
├── requirements.txt                  # Python dependencies
├── bovine_breeds_metadata.csv        # Current source metadata index
├── Indian_bovine_breeds/             # Original raw image tree; preserve as source data
│   └── Indian_bovine_breeds/
│       └── <breed>/                  # One directory per class label
├── data/
│   ├── raw/
│   │   └── images/                   # Canonical future location for untouched images
│   ├── external/                     # External datasets or downloaded references
│   ├── metadata/                     # Versioned metadata copies and schemas
│   ├── processed/                    # Resized, cleaned, or augmented images
│   └── splits/                       # Train/validation/test manifests
├── src/
│   ├── data/                         # Loaders, validation, and preprocessing
│   ├── features/                     # Feature extraction and transforms
│   ├── training/                     # Training entry points and configuration
│   └── evaluation/                   # Metrics, confusion matrices, and evaluation
├── configs/                          # YAML/JSON experiment configurations
├── notebooks/                        # Exploratory or experiment notebooks
├── models/                           # Exported model definitions or legacy model code
├── artifacts/
│   └── checkpoints/                  # Generated model checkpoints
├── reports/                          # Generated audit, metric, and visualization reports
└── tests/                            # Automated tests
```

## Data Flow

```text
raw images + metadata
        -> audit and path validation
        -> processed images and split manifests
        -> training
        -> checkpoints and evaluation reports
```

## Naming and Ownership Rules

- `Indian_bovine_breeds/` is source data. Do not edit individual images in place.
- `bovine_breeds_metadata.csv` is the current metadata source. Any normalized copy belongs in `data/metadata/`.
- Generated files belong in `data/processed/`, `data/splits/`, `artifacts/`, or `reports/`, never beside source images.
- A breed directory name is the class label after normalization of separators and whitespace.
- Paths in metadata are relative to the dataset root used by the script, not absolute machine-specific paths.
- Use forward slashes in new metadata files for portability, even on Windows.
- Keep train, validation, and test manifests separate and record the random seed used to create them.

## AI Assistant Reading Order

When another AI assistant receives this repository, it should read:

1. `PROJECT_ARCHITECTURE.md`
2. `DATASET_CARD.md`
3. `data/README.md`
4. `audit_dataset.py`
5. `bovine_breeds_metadata.csv` header and a small sample

It should then inspect the requested module and run the narrowest relevant test or audit command before making changes.

## Current State and Known Boundaries

The source tree, metadata file, and audit script already exist. The new `data/` directories establish the intended future layout but do not silently move or duplicate the large image dataset. A migration should update the metadata paths and audit configuration together, then run the audit before deleting any legacy path.
