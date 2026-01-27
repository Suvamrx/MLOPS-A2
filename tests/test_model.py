import torch
import torch.nn as nn
from src.train import SimpleCNN

def test_model_forward():
    model = SimpleCNN()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
    assert torch.is_tensor(out)
