import pytest


def test_slice_attention_respects_mask_shape():
    torch = pytest.importorskip("torch")
    from orion.models.necks.slice_aggregator import SliceAttentionAggregator

    output = SliceAttentionAggregator(4)(
        torch.randn(2, 3, 4),
        torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
    )
    assert output.shape == (2, 4)
