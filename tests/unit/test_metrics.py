import numpy as np
from orion.evaluation.metrics import calculate_metrics


def test_metrics_ignore_unknown_labels():
    result = calculate_metrics(
        np.array([[1, -1], [0, 1], [1, 0]]),
        np.array([[0.9, 0.5], [0.1, 0.8], [0.8, 0.2]]),
        ["a", "b"],
    )
    assert result["macro_auc_roc"] == 1.0
