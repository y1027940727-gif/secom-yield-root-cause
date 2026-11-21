import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.pipeline import (select_sensor_columns, expanding_window_splits, bootstrap_ci,
                          evaluate, to_markdown)


def test_sensor_selection_uses_training_period_only():
    # high_missing_in_test is complete in train, missing only in the future -> must be KEPT
    # dead_in_train has no variation in train but varies in the future -> must be DROPPED
    train = pd.DataFrame({
        "ok": [1.0, 2.0, 3.0, 4.0],
        "high_missing_in_test": [1.0, 2.0, 3.0, 4.0],
        "dead_in_train": [5.0, 5.0, 5.0, 5.0],
        "high_missing_in_train": [np.nan, np.nan, np.nan, 1.0],
    })
    assert select_sensor_columns(train) == ["ok", "high_missing_in_test"]


def test_expanding_window_splits_are_chronological_and_disjoint():
    splits = list(expanding_window_splits(100, n_folds=4, initial_train_frac=0.5))
    assert len(splits) == 4
    prev_train = 0
    for tr, te in splits:
        assert tr.max() < te.min()               # train strictly before test
        assert len(set(tr) & set(te)) == 0
        assert len(tr) > prev_train              # training window expands
        prev_train = len(tr)
    assert splits[-1][1].max() == 99             # last fold ends at the last row


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y * 0.5 + rng.random(300)
    lo, hi = bootstrap_ci(y, s, roc_auc_score, n_boot=300)
    assert lo < hi
    assert lo <= roc_auc_score(y, s) <= hi


def test_evaluate_runs_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"Sensor_{i}" for i in range(8)])
    X["Sensor_dead"] = 1.0
    y = pd.Series((X["Sensor_0"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int))
    factory = lambda w: {"LR": LogisticRegression(max_iter=200)}
    res = evaluate(X, y, models_factory=factory, use_shap=False)
    r = res["pooled_out_of_fold"]["LR"]
    assert r["roc_auc"] > 0.7                      # real signal is detected
    assert r["roc_auc_ci95"][0] < r["roc_auc"] < r["roc_auc_ci95"][1]
    assert abs(r["baseline_pr_auc"] - y.iloc[200:].mean()) < 1e-9
    assert "LR" in to_markdown(res)
