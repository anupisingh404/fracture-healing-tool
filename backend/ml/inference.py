from __future__ import annotations
import numpy as np
from backend.ml.pipeline import MLPipeline, LABEL_INV

_WEIGHTS = {
    "RandomForest": 0.15,
    "XGBoost": 0.35,
    "LogisticRegression": 0.15,
    "SVM": 0.15,
    "GradientBoosting": 0.20,
}


def run_inference(pipeline: MLPipeline, features: np.ndarray) -> dict:
    """
    Returns per-model class probabilities and an ensemble score.
    Probability reported is P(Good) — index 2 in [Poor, Moderate, Good].
    """
    X_scaled = pipeline.scaler.transform(features.reshape(1, -1))
    all_probs: dict[str, dict] = {}
    good_probs: dict[str, float] = {}

    for name, clf in pipeline.models.items():
        proba = clf.predict_proba(X_scaled)[0]  # shape: (3,)
        all_probs[name] = {
            LABEL_INV.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(proba)
        }
        good_probs[name] = round(float(proba[2]), 4)

    best_prob = good_probs.get(pipeline.best_model_name, 0.5)

    total_w = sum(_WEIGHTS.get(n, 0.2) for n in pipeline.models)
    ensemble = sum(
        good_probs[n] * _WEIGHTS.get(n, 0.2) for n in pipeline.models
    ) / total_w

    # Predicted class from best model
    X_cls = pipeline.scaler.transform(features.reshape(1, -1))
    best_clf = pipeline.models[pipeline.best_model_name]
    pred_label_idx = int(best_clf.predict(X_cls)[0])
    predicted_category = LABEL_INV.get(pred_label_idx, "Moderate")

    return {
        "probability": round(best_prob, 4),
        "best_model": pipeline.best_model_name,
        "good_probs": good_probs,
        "all_probs": all_probs,
        "ensemble_probability": round(ensemble, 4),
        "predicted_category": predicted_category,
    }
