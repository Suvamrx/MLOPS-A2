import pytest
from src.preprocess import augment_image
from PIL import Image
import numpy as np

def test_augment_image_flip():
    # Create a dummy image (RGB, 224x224)
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
    flipped = augment_image(img)
    assert isinstance(flipped, Image.Image)
    assert flipped.size == (224, 224)

def test_augment_image_identity():
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    out = augment_image(img)
    assert isinstance(out, Image.Image)
    assert out.size == (224, 224)
