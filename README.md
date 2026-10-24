
# Indian Cattle & Buffalo Breed Recognition

An image-classification application that predicts cattle and buffalo breeds using a fine-tuned EfficientNet-B3 model and Streamlit.

## Features

- Upload an image in JPG, JPEG, or PNG format
- Capture an image directly using the device camera
- Predict the most likely breed among 41 classes
- Display confidence and top-3 breed predictions
- Preliminary image validation before breed classification

## Model

- Architecture: EfficientNet-B3
- Best validation accuracy: 68.68%
- Test accuracy: 67.18%
- Test macro F1-score: 63.66%
- Number of classes: 41
- Best checkpoint: `model/efficientnet_b3_best.pth`

These metrics are from the existing evaluation run. Performance on individual images may vary. The preliminary image validator is heuristic and is not guaranteed to reject every unrelated image.

## Project Structure

```text
project/
├── app.py
├── model/
│   └── efficientnet_b3_best.pth
├── configs/
├── data/
├── notebooks/
├── reports/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Python 3.12 is recommended for this environment.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

From the project root directory:

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## Dataset

The project uses the Indian Bovine Breeds image dataset.

The original image dataset is not included in this repository. See `DATASET_CARD.md` for dataset details and source information.

## Model Checkpoint

The trained checkpoint is expected at:

`model/efficientnet_b3_best.pth`

The checkpoint stores the model weights, class names, and class-to-index mapping.

## Limitations

- Predictions depend on image quality, lighting, and breed similarity.
- Confidence scores are model estimates, not guarantees.
- The image validator is a preliminary heuristic and may produce false positives or false negatives.
- The model's reported test accuracy is 67.18%; predictions are not always correct.

## License and Dataset Attribution

Add the applicable project license and dataset attribution after confirming the source dataset's usage terms.
