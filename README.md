# Semiconductor Wafer Yield Anomaly Detection & SHAP Root-Cause Analysis

A time-ordered, leakage-safe machine-learning pipeline on the public UCI SECOM dataset (1,567 wafers, 590 sensors, 104 yield failures, 6.6%). The goal is to see whether sensor readings predict later yield failures, and which sensors are candidates for engineering review.

## Method
1. **Chronological order**: wafers are sorted by timestamp.
2. **Expanding-window validation**: train on everything before each future block (4 blocks after the first 50% of the data), so the model never sees the future.
3. **Training-period-only preprocessing**: sensor filtering (drop sensors with more than 50% missing values or zero variance), median imputation and standardization are all fit on the training window of each fold.
4. **Models**: L1 logistic regression, random forest, XGBoost (class-imbalance weighting).
5. **Uncertainty**: 95% bootstrap confidence intervals on pooled out-of-fold predictions, compared with naive baselines (random ROC-AUC = 0.5; PR-AUC = failure rate).
6. **Candidate sensors**: SHAP top-15 sensors per model; sensors in all three models' top 15 are counted across folds to show how stable they are.

## Results
Run the pipeline to regenerate `results.md` and `results.json`:

```bash
pip install pandas numpy scikit-learn xgboost shap pytest
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. python src/pipeline.py   # downloads SECOM from UCI
```

_Results table: paste the contents of `results.md` here after running._

### Earlier version (v1), for reference
v1 used a single 75/25 chronological split and filtered sensors on the full dataset (a mild leakage that v2 removes). Its test ROC-AUC was 0.519-0.632 and only 2 sensors (`Sensor_0`, `Sensor_59`) were shared across the three models' top 15.

## Limitations
- The predictive signal is weak; models near ROC-AUC 0.5 are not evidence of a real effect.
- Sensors shared by the models are **candidates for engineering review, not confirmed root causes**.
- Only about 100 failures in total, so every estimate is noisy (see the confidence intervals).
- No hyperparameter search.

## Tests
`pytest` covers: sensor selection uses only the training period, expanding-window splits are chronological and disjoint, bootstrap intervals are ordered and contain the estimate, and an end-to-end run on synthetic data detects a planted signal.
