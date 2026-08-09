import numpy as np
import pytest

from orion.data.splits.leakage import assert_no_group_leakage
from orion.data.splits.stratified import iterative_group_kfold


def test_iterative_folds_do_not_leak_groups():
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]])
    groups = np.array(["a", "a", "b", "c", "d", "e"])
    for train, valid in iterative_group_kfold(labels, groups, 3):
        assert_no_group_leakage(groups[train], groups[valid])


def test_leakage_check_fails():
    with pytest.raises(ValueError):
        assert_no_group_leakage(["p1"], ["p1"])
