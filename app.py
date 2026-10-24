
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "efficientnet_b3_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load the trained model only once
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]

    if len(class_names) != 41 or len(class_to_idx) != 41:
        raise ValueError("Expected exactly 41 breed classes in checkpoint.")

    # Confirm that the saved class mapping matches the output indices.
    for idx, class_name in enumerate(class_names):
        if class_to_idx.get(class_name) != idx:
            raise ValueError(
                f"Class mapping mismatch for {class_name}"
            )

    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, checkpoint




# --------------------------------------------------
# Input image validation model
# --------------------------------------------------
@st.cache_resource
def load_image_validator():
    weights = models.ResNet50_Weights.DEFAULT
    validator = models.resnet50(weights=weights)
    validator.eval()
    validator.to(DEVICE)

    return validator, weights, weights.transforms()


def validate_bovine_image(image, validator, weights, validator_transform):
    """
    Preliminary check using ImageNet labels.
    This is a heuristic, not a guaranteed animal detector.
    """
    animal_terms = (
        "ox",
        "water buffalo",
        "bison",
        "cow",
        "bull",
        "cattle",
        "buffalo",
    )

    tensor = validator_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = validator(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=5)
    labels = weights.meta["categories"]

    top_labels = [
        (labels[idx.item()].lower(), prob.item())
        for prob, idx in zip(top_probs, top_indices)
    ]

    matched = [
        (label, score)
        for label, score in top_labels
        if any(term in label for term in animal_terms)
    ]

    # Conservative preliminary decision; tune only after testing.
    accepted = any(score >= 0.05 for _, score in matched)

    return accepted, top_labels


# --------------------------------------------------
# Preprocessing: match validation inference pipeline
# --------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="Indian Bovine Breed Recognition",
    page_icon="🐄",
    layout="centered",
)

st.title("🐄 Indian Cattle & Buffalo Breed Recognition")
st.write(
    "Upload a cattle or buffalo image to predict its breed "
    "using our trained EfficientNet-B3 model."
)

st.caption("41 breed classes | EfficientNet-B3 | Image size: 300 × 300")

try:
    model, class_names, checkpoint = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

st.success("Trained model loaded successfully.")

try:
    validator, validator_weights, validator_transform = (
        load_image_validator()
    )
    st.success("Image validator loaded successfully.")
except Exception as exc:
    st.error(f"Could not load image validator: {exc}")
    st.stop()


input_method = st.radio(
    "Choose image input method",
    ["Upload Image", "Capture from Camera"],
    horizontal=True,
)

uploaded_file = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a cattle or buffalo image",
        type=["jpg", "jpeg", "png"],
        key="breed_image_upload",
    )
else:
    uploaded_file = st.camera_input(
        "Take a photo of the cattle or buffalo",
        key="breed_camera_input",
    )


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        if st.button("Predict Breed", type="primary"):

            
            is_valid, validator_labels = validate_bovine_image(
                image,
                validator,
                validator_weights,
                validator_transform,
            )

            if not is_valid:
                st.error(
                    "Invalid image detected. Please upload or capture "
                    "a clear cattle or buffalo image."
                )
                st.caption(
                    "Image validator ke top predictions: "
                    + ", ".join(
                        f"{label} ({score * 100:.1f}%)"
                        for label, score in validator_labels[:3]
                    )
                )
                st.stop()


            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            top_probs, top_indices = torch.topk(
                probabilities,
                k=min(3, len(class_names)),
            )

            predicted_idx = top_indices[0].item()
            predicted_breed = class_names[predicted_idx]
            confidence = top_probs[0].item() * 100

            st.subheader("Prediction Result")
            st.metric("Predicted Breed", predicted_breed)
            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.subheader("Top 3 Predicted Breeds")

            for rank, (prob, idx) in enumerate(
                zip(top_probs.tolist(), top_indices.tolist()),
                start=1,
            ):
                breed = class_names[idx]
                score = prob * 100

                st.write(f"**{rank}. {breed} — {score:.2f}%**")
                st.progress(float(prob))

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

with st.expander("Model Information"):
    st.write("**Architecture:** EfficientNet-B3")
    st.write(f"**Best training epoch:** {checkpoint['best_epoch']}")
    st.write(
        "**Best validation accuracy:** "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )
    st.write(f"**Number of classes:** {len(class_names)}")
    st.write(f"**Device:** {DEVICE}")



