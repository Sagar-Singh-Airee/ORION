from __future__ import annotations
def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.5f}" for key, value in metrics.items())
