"""Feature aggregation layers."""
from .attention_pool import AttentionPool
from .slice_aggregator import MaskedMeanAggregator, SliceAttentionAggregator

__all__ = ["AttentionPool", "MaskedMeanAggregator", "SliceAttentionAggregator"]
