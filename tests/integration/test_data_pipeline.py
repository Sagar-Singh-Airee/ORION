import numpy as np

from orion.data.splits.stratified import iterative_group_kfold


def test_fold_generation_covers_every_record_once():
    labels = np.array([[0], [1], [0], [1], [0], [1]])
    splits = iterative_group_kfold(labels, np.arange(len(labels)), 3)
    assert sorted(np.concatenate([valid for _, valid in splits]).tolist()) == list(range(len(labels)))
