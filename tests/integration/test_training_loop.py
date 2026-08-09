import pytest


def test_training_components_import_when_torch_is_available():
    pytest.importorskip("torch")
    from orion.training.trainer import Trainer

    assert Trainer is not None
