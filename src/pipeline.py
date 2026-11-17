import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

def clean_sensor_features(X_raw: pd.DataFrame, max_missing: float = 0.50, min_std: float = 1e-6) -> pd.DataFrame:
    """Drop high-missingness and zero-variance dead sensors."""
    missing_ratio = X_raw.isnull().mean()
    X_valid = X_raw[missing_ratio[missing_ratio <= max_missing].index]
    std_vals = X_valid.std()
    return X_valid[std_vals[std_vals > min_std].index]

def time_based_split(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series, train_ratio: float = 0.75):
    """Sort chronologically and split to prevent future-lot leakage."""
    sort_idx = np.argsort(timestamps)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    split_pt = int(len(X_sorted) * train_ratio)
    return X_sorted.iloc[:split_pt], X_sorted.iloc[split_pt:], y_sorted.iloc[:split_pt], y_sorted.iloc[split_pt:]

def run_benchmark():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    labels_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    X_raw = pd.read_csv(data_url, sep=r'\s+', header=None)
    X_raw.columns = [f"Sensor_{i}" for i in range(X_raw.shape[1])]
    labels_df = pd.read_csv(labels_url, sep=r'\s+', header=None, names=["label", "timestamp"])
    y_raw = (labels_df["label"] == 1).astype(int)
    timestamps = pd.to_datetime(labels_df["timestamp"], dayfirst=True)

    X_clean = clean_sensor_features(X_raw)
    X_train_raw, X_test_raw, y_train, y_test = time_based_split(X_clean, y_raw, timestamps)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_train_raw)), columns=X_clean.columns)
    X_test = pd.DataFrame(scaler.transform(imputer.transform(X_test_raw)), columns=X_clean.columns)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "Logistic Regression (L1)": LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42)
    }

    results, top_features = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "roc_auc": round(roc_auc_score(y_test, y_prob), 3),
            "pr_auc": round(average_precision_score(y_test, y_prob), 3)
        }
        if "Logistic" in name:
            shap_vals = shap.LinearExplainer(model, X_train).shap_values(X_test)
        elif "Random Forest" in name:
            sv = shap.TreeExplainer(model).shap_values(X_test)
            shap_vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if len(sv.shape) == 3 else sv)
        else:
            shap_vals = shap.TreeExplainer(model).shap_values(X_test)
        top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:15]
        top_features[name] = set(X_clean.columns[top_idx])

    consensus = sorted(list(set.intersection(*top_features.values())))
    return results, consensus

if __name__ == "__main__":
    res, sensors = run_benchmark()
    print("Benchmark Results:", res)
    print("Consensus Root-Cause Sensors:", sensors)
