import numpy as np

from orion.data.transforms.intensity import gamma_correct
from orion.data.transforms.spatial import horizontal_flip


def test_spatial_and_intensity_transforms_preserve_shape():
    volume = np.arange(12, dtype=np.float32).reshape(1, 3, 4) / 11
    assert horizontal_flip(volume).shape == volume.shape
    assert gamma_correct(volume, 1.2).shape == volume.shape
