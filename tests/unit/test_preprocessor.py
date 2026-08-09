import numpy as np

from orion.data.dicom.preprocessor import normalize_intensity, select_slice_indices


def test_uniform_slice_selection_has_expected_size():
    assert len(select_slice_indices(100, 24)) == 24


def test_constant_volume_normalizes_to_zero():
    assert np.all(normalize_intensity(np.ones((2, 3, 3))) == 0)
