import torch

from src.liveness.anti_spoof.model import AntiSpoofCNN


model = AntiSpoofCNN()

dummy_input = torch.randn(1, 3, 224, 224)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Anti-Spoof CNN is working.")