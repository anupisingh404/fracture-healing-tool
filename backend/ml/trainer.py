"""
Standalone training script.
Usage: python3 -m backend.ml.trainer
"""
from __future__ import annotations
import numpy as np
import logging
import os
import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "RandomForest",
    "XGBoost",
    "LogisticRegression",
    "SVM",
    "GradientBoosting",
]

_MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    # "XGBoost": XGBClassifier(
    #     n_estimators=200, max_depth=6, learning_rate=0.05,
    #     subsample=0.8, colsample_bytree=0.8,
    #     eval_metric="mlogloss", random_state=42,
    #     verbosity=0,
    # ),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    ),
    "SVM": SVC(
        C=1.0, kernel="rbf", probability=True,
        class_weight="balanced", random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    ),
}


def train_all(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
) -> tuple[dict, dict, str]:
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained: dict = {}
    scores: dict = {}

    for name, clf in _MODELS.items():
        cv_score = cross_val_score(
            clf, X_scaled, y, cv=cv, scoring="f1_macro", n_jobs=-1
        ).mean()
        clf.fit(X_scaled, y)
        trained[name] = clf
        scores[name] = round(float(cv_score), 4)
        logger.info(f"  {name}: CV f1_macro = {scores[name]:.4f}")

    best = max(scores, key=scores.get)
    logger.info(f"Best model: {best} ({scores[best]:.4f})")
    return trained, scores, best


if __name__ == "__main__":
    import pandas as pd
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = os.path.join(os.path.dirname(__file__), "../../")
    csv_path = os.path.join(root, "data/sample_patients.csv")
    if not os.path.exists(csv_path):
        print("ERROR: sample_patients.csv not found. Run: python3 data/generate_synthetic.py")
        sys.exit(1)

    from backend.ml.pipeline import MLPipeline, LABEL_MAP
    pipe = MLPipeline()
    df = pd.read_csv(csv_path)
    X = pipe.featurize_df(df)
    y = df["healing_category"].map(LABEL_MAP).values

    logger.info(f"Training on {len(X)} samples …")
    trained, scores, best = train_all(X, y, pipe.scaler)
    pipe.models = trained
    pipe.cv_scores = scores
    pipe.best_model_name = best
    pipe._save_models()
    print("\nCV scores:", scores)
    print("Best model:", best)
