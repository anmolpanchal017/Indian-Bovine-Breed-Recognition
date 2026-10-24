
import torch

from src.training.model import create_model

model = create_model(num_classes=41, pretrained=False)
model.eval()

dummy_input = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("Input shape:", tuple(dummy_input.shape))
print("Output shape:", tuple(output.shape))
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
print("Model test passed!")

