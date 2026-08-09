import pytest


def test_config_merges_default_and_experiment():
    pytest.importorskip("omegaconf")
    from orion.utils.config import load_config

    cfg = load_config("experiment/baseline_resnet50.yaml")
    assert cfg.labels.num_classes == 12
    assert cfg.model.vision_backbone.name == "resnet50"
