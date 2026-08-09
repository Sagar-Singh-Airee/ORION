import numpy as np

from orion.evaluation.metrics import calculate_metrics
from orion.inference.submission import create_submission


def test_prediction_metrics_to_submission_smoke():
    targets = np.array([[0, 1], [1, 0]])
    probabilities = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert calculate_metrics(targets, probabilities, ["a", "b"])["macro_auc"] == 1.0
    assert len(create_submission(["a", "b"], probabilities, ["a", "b"])) == 2
