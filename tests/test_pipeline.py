import pandas as pd
import numpy as np
from src.pipeline import clean_sensor_features, time_based_split

def test_clean_sensor_features_drops_high_missing_and_dead_sensors():
    df = pd.DataFrame({
        "valid_sensor": [1.0, 2.0, 3.0, 4.0],
        "high_missing": [np.nan, np.nan, np.nan, 1.0],  # 75% missing
        "dead_sensor": [5.0, 5.0, 5.0, 5.0]            # 0 variance
    })
    cleaned = clean_sensor_features(df, max_missing=0.50)
    assert list(cleaned.columns) == ["valid_sensor"]

def test_time_based_split_prevents_future_leakage():
    X = pd.DataFrame({"f1": [10, 20, 30, 40]})
    y = pd.Series([0, 1, 0, 1])
    # Unordered timestamps
    ts = pd.to_datetime(["2026-01-04", "2026-01-01", "2026-01-03", "2026-01-02"])
    X_tr, X_te, y_tr, y_te = time_based_split(X, y, ts, train_ratio=0.5)
    # Earliest two records (f1=20, 40) must be in train; latest two (30, 10) in test
    assert list(X_tr["f1"]) == [20, 40]
    assert list(X_te["f1"]) == [30, 10]
