import numpy as np
import pytest

from orion.data.splits.leakage import assert_no_group_leakage
from orion.data.splits.stratified import iterative_group_kfold
from orion.evaluation.metrics import calculate_metrics
from orion.inference.ensemble import ensemble_predictions
from orion.inference.submission import create_submission
from orion.weak_supervision.label_model import train_label_model


def test_metrics_ignore_unknown_weak_labels():
    targets = np.array([[1, -1], [0, 1], [1, 0]])
    probabilities = np.array([[0.9, 0.4], [0.1, 0.8], [0.8, 0.1]])
    metrics = calculate_metrics(targets, probabilities, ["a", "b"])
    assert metrics["macro_auc_roc"] == pytest.approx(1.0)


def test_group_folds_never_leak_a_patient():
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]])
    groups = np.array(["a", "a", "b", "c", "d", "e"])
    for train, valid in iterative_group_kfold(labels, groups, n_splits=3):
        assert_no_group_leakage(groups[train], groups[valid])


def test_ensemble_and_submission_are_strict():
    predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert np.allclose(ensemble_predictions([predictions, predictions]), predictions)
    submission = create_submission(["one", "two"], predictions, ["first", "second"])
    assert list(submission) == ["study_id", "first", "second"]
    with pytest.raises(ValueError, match="unique"):
        create_submission(["one", "one"], predictions, ["first", "second"])


def test_weak_label_fallback_preserves_abstention_uncertainty():
    labels = np.array([[1, 1, -1], [0, -1, 0], [-1, -1, -1]])
    probabilities = train_label_model(labels)
    assert probabilities.shape == (3, 2)
    assert probabilities[0, 1] > 0.5
    assert probabilities[1, 1] < 0.5
    assert probabilities[2, 1] == pytest.approx(0.5)
