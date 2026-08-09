import numpy as np

from orion.inference.ensemble import ensemble_predictions
from orion.inference.submission import create_submission


def test_ensemble_output_can_be_submitted():
    prediction = np.array([[0.1, 0.9], [0.8, 0.2]])
    ensemble = ensemble_predictions([prediction, prediction])
    submission = create_submission(["one", "two"], ensemble, ["a", "b"])
    assert submission.shape == (2, 3)
