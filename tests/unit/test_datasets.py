import pytest


def test_dataset_requires_real_metadata_when_no_records():
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from orion.data.datasets.knee_mri import KneeMRIDataset

    with pytest.raises(FileNotFoundError):
        KneeMRIDataset(OmegaConf.create({"data": {"root": "/missing"}}))
