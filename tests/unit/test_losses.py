import pytest


def test_masked_bce_accepts_unknown_targets():
    torch = pytest.importorskip("torch")
    from orion.models.losses.bce import MaskedBCEWithLogitsLoss

    result = MaskedBCEWithLogitsLoss()(torch.zeros(2, 2), torch.tensor([[1.0, -1.0], [0.0, 1.0]]))
    assert torch.isfinite(result)
