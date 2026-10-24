# Dataset Card

## Dataset Name

Indian Bovine Breeds Image Dataset

## Task

Multiclass image classification: predict the bovine breed represented by an image.

## Source Layout

Images are currently stored under:

```text
Indian_bovine_breeds/Indian_bovine_breeds/<breed>/<image-file>
```

There are 40 breed directories in the current source tree. The directory name is the intended class label.

## Metadata Contract

The current file is `bovine_breeds_metadata.csv`.

| Column | Meaning | Required |
|---|---|---|
| `image_id` | Image filename or unique image identifier | Yes |
| `breed` | Target class label | Yes |
| `path` | Path to the image relative to the configured dataset root | Yes |
| `file extension` | File extension recorded in the metadata | Informational |

The metadata must contain one row per image. `path` must resolve to an existing readable image. The folder component of `path` must match `breed`, ignoring case and surrounding whitespace.

## Data Quality Checks

`audit_dataset.py` checks:

- required metadata columns
- total records and unique breeds
- duplicate paths
- missing files
- unreadable images
- image extensions
- folder-label mismatches

It writes `dataset_audit_report.csv` at the project root. This is a generated report and should not be treated as source metadata.

## Limitations

The repository does not yet document image provenance, licensing, capture conditions, class balance, subject identity overlap, or demographic/geographic coverage. Model performance must not be interpreted as a guarantee of breed identity in uncontrolled field conditions.

## Reproducibility Requirements

Every future training run should record the metadata version, preprocessing settings, split seed, class mapping, Python environment, and model configuration in `configs/`, `reports/`, or the relevant experiment notebook.
