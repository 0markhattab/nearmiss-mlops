#!/usr/bin/env python3
import argparse, os, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
try:
    import mlflow
    MLFLOW = True
except Exception:
    MLFLOW = False

def main(train_csv: str, out_dir: str, test_size: float, random_state: int):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(train_csv)
    # Features & label
    label_col = "near_miss"
    feature_cols = [c for c in df.columns if c not in {"event_id","store_id","ts", label_col}]
    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    base = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    grid = [
        {**base, "num_leaves": 15, "min_child_samples": 40},
        {**base, "num_leaves": 31, "min_child_samples": 30},
        {**base, "num_leaves": 63, "min_child_samples": 20}
    ]

    best, best_score, best_params = None, -1.0, None
    for params in grid:
        model = LGBMClassifier(**params)
        clf = CalibratedClassifierCV(model, method="isotonic", cv=3)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:,1]
        auprc = average_precision_score(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        if MLFLOW:
            mlflow.log_metric("val_auprc", auprc)
            mlflow.log_metric("val_auc", auc)
        if auprc > best_score:
            best_score, best = auprc, clf
            best_params = params

    model_path = os.path.join(out_dir, "lightgbm_calibrated.pkl")
    joblib.dump(best, model_path)

    yhat = best.predict(X_test)
    proba = best.predict_proba(X_test)[:,1]
    metrics = {
        "best_params": best_params,
        "auprc": float(average_precision_score(y_test, proba)),
        "auc": float(roc_auc_score(y_test, proba)),
        "report": classification_report(y_test, yhat, output_dict=True)
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", model_path)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()
    main(args.train_csv, args.out_dir, args.test_size, args.random_state)
