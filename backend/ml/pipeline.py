from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.preprocessing import StandardScaler

from backend.schemas.patient import PatientInput, BiomarkerTrends, HealingCategory

logger = logging.getLogger(__name__)

GENDER_MAP = {"male": 0, "female": 1, "other": 2}
LOCATION_MAP = {
    "femur": 0, "tibia": 1, "radius": 2, "humerus": 3,
    "ulna": 4, "fibula": 5, "pelvis": 6, "vertebra": 7,
}
LABEL_MAP = {"Poor": 0, "Moderate": 1, "Good": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}

MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/sample_patients.csv")


class MLPipeline:
    def __init__(self, best_model_name: str = "XGBoost"):
        self.scaler = StandardScaler()
        self.models: dict = {}
        self.best_model_name = best_model_name
        self.cv_scores: dict = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def featurize(self, patient: PatientInput) -> tuple[np.ndarray, list[str]]:
        p = patient
        gender_enc = GENDER_MAP.get(p.gender.value, 2)
        loc_enc = LOCATION_MAP.get(p.fracture_location.value, 0)

        bsap = [p.biomarkers_day1.bsap, p.biomarkers_week3.bsap, p.biomarkers_week6.bsap]
        alp = [p.biomarkers_day1.alp, p.biomarkers_week3.alp, p.biomarkers_week6.alp]
        p1np = [p.biomarkers_day1.p1np, p.biomarkers_week3.p1np, p.biomarkers_week6.p1np]
        ca = [p.minerals_day1.calcium, p.minerals_week3.calcium, p.minerals_week6.calcium]
        phos = [p.minerals_day1.phosphorus, p.minerals_week3.phosphorus, p.minerals_week6.phosphorus]
        callus = [p.callus_d1, p.callus_w3, p.callus_w6]

        def delta(s): return s[-1] - s[0]

        bsap_alp_ratio = bsap[2] / (alp[2] + 1e-9)
        ca_phos_product = ca[2] * phos[2]
        callus_growth_rate = (callus[2] - callus[0]) / 42.0  # per day (6 weeks)

        features = [
            p.age, gender_enc, loc_enc,
            *bsap, *alp, *p1np,
            *ca, *phos,
            *callus,
            delta(bsap), delta(alp), delta(p1np),
            delta(ca), delta(phos),
            callus[1] - callus[0], callus[2] - callus[1], delta(callus),
            bsap_alp_ratio, ca_phos_product, callus_growth_rate,
        ]

        names = [
            "age", "gender", "fracture_location",
            "bsap_d1", "bsap_w3", "bsap_w6",
            "alp_d1", "alp_w3", "alp_w6",
            "p1np_d1", "p1np_w3", "p1np_w6",
            "ca_d1", "ca_w3", "ca_w6",
            "phos_d1", "phos_w3", "phos_w6",
            "callus_d1", "callus_w3", "callus_w6",
            "bsap_delta", "alp_delta", "p1np_delta",
            "ca_delta", "phos_delta",
            "callus_d1_w3", "callus_w3_w6", "callus_d1_w6",
            "bsap_alp_ratio_w6", "ca_phos_product_w6", "callus_growth_rate",
        ]

        return np.array(features, dtype=float), names

    def featurize_df(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised featurize for a DataFrame of raw CSV rows."""
        def _delta(a, b): return df[b] - df[a]

        X = np.column_stack([
            df["age"],
            df["gender"].map(GENDER_MAP).fillna(2),
            df["fracture_location"].map(LOCATION_MAP).fillna(0),
            df["bsap_d1"], df["bsap_w3"], df["bsap_w6"],
            df["alp_d1"], df["alp_w3"], df["alp_w6"],
            df["p1np_d1"], df["p1np_w3"], df["p1np_w6"],
            df["ca_d1"], df["ca_w3"], df["ca_w6"],
            df["phos_d1"], df["phos_w3"], df["phos_w6"],
            df["callus_d1"], df["callus_w3"], df["callus_w6"],
            _delta("bsap_d1", "bsap_w6"),
            _delta("alp_d1", "alp_w6"),
            _delta("p1np_d1", "p1np_w6"),
            _delta("ca_d1", "ca_w6"),
            _delta("phos_d1", "phos_w6"),
            _delta("callus_d1", "callus_w3"),
            _delta("callus_w3", "callus_w6"),
            _delta("callus_d1", "callus_w6"),
            df["bsap_w6"] / (df["alp_w6"] + 1e-9),
            df["ca_w6"] * df["phos_w6"],
            (df["callus_w6"] - df["callus_d1"]) / 42.0,
        ])
        return X.astype(float)

    # ------------------------------------------------------------------
    # Load / train
    # ------------------------------------------------------------------

    def load_or_train(self):
        scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.pkl")
        if os.path.exists(scaler_path) and len(os.listdir(MODEL_SAVE_DIR)) > 1:
            self._load_models()
        else:
            logger.info("No saved models found — training from synthetic data …")
            self._train_from_csv()

    def _load_models(self):
        from backend.ml.trainer import MODEL_NAMES
        self.scaler = joblib.load(os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
        for name in MODEL_NAMES:
            path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
        scores_path = os.path.join(MODEL_SAVE_DIR, "cv_scores.pkl")
        if os.path.exists(scores_path):
            self.cv_scores = joblib.load(scores_path)
        logger.info(f"Loaded {len(self.models)} models from disk.")

    def _train_from_csv(self):
        from backend.ml.trainer import train_all
        csv = os.path.abspath(DATA_PATH)
        if not os.path.exists(csv):
            raise FileNotFoundError(
                f"Training data not found at {csv}. "
                "Run: python3 data/generate_synthetic.py"
            )
        df = pd.read_csv(csv)
        X = self.featurize_df(df)
        y = df["healing_category"].map(LABEL_MAP).values
        self.models, self.cv_scores, self.best_model_name = train_all(X, y, self.scaler)
        self._save_models()

    def _save_models(self):
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
        joblib.dump(self.cv_scores, os.path.join(MODEL_SAVE_DIR, "cv_scores.pkl"))
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(MODEL_SAVE_DIR, f"{name}.pkl"))
        logger.info("Models saved to disk.")


# ------------------------------------------------------------------
# Trend computation
# ------------------------------------------------------------------

def _pct_change(series: list[float]) -> float:
    return round((series[-1] - series[0]) / (series[0] + 1e-9) * 100, 2)


def _trend_narrative(bsap, alp, p1np, callus) -> str:
    parts = []
    if bsap[-1] > bsap[0] * 1.1:
        parts.append(f"BSAP rising (+{_pct_change(bsap):.0f}%), suggesting active bone formation")
    elif bsap[-1] < bsap[0] * 0.9:
        parts.append(f"BSAP declining ({_pct_change(bsap):.0f}%), reduced osteoblast activity")
    else:
        parts.append("BSAP stable, moderate osteoblast activity")

    if callus[-1] > callus[0] * 1.5:
        parts.append(f"callus growing well (+{_pct_change(callus):.0f}%)")
    elif callus[-1] < callus[0] * 1.1:
        parts.append("callus growth minimal — watch for delayed union")

    return "; ".join(parts).capitalize() + "."


def compute_trends(patient: PatientInput) -> BiomarkerTrends:
    p = patient
    bsap = [p.biomarkers_day1.bsap, p.biomarkers_week3.bsap, p.biomarkers_week6.bsap]
    alp = [p.biomarkers_day1.alp, p.biomarkers_week3.alp, p.biomarkers_week6.alp]
    p1np = [p.biomarkers_day1.p1np, p.biomarkers_week3.p1np, p.biomarkers_week6.p1np]
    ca = [p.minerals_day1.calcium, p.minerals_week3.calcium, p.minerals_week6.calcium]
    phos = [p.minerals_day1.phosphorus, p.minerals_week3.phosphorus, p.minerals_week6.phosphorus]
    callus = [p.callus_d1, p.callus_w3, p.callus_w6]

    return BiomarkerTrends(
        bsap_trend=bsap,
        alp_trend=alp,
        p1np_trend=p1np,
        ca_trend=ca,
        phos_trend=phos,
        callus_trend=callus,
        bsap_delta_pct=_pct_change(bsap),
        alp_delta_pct=_pct_change(alp),
        p1np_delta_pct=_pct_change(p1np),
        callus_delta_pct=_pct_change(callus),
        trend_summary=_trend_narrative(bsap, alp, p1np, callus),
    )


def classify_category(callus_w6: float) -> HealingCategory:
    if callus_w6 < 100:
        return HealingCategory.POOR
    if callus_w6 <= 180:
        return HealingCategory.MODERATE
    return HealingCategory.GOOD


def extract_risk_flags(trends: BiomarkerTrends) -> list[str]:
    flags = []
    if trends.bsap_delta_pct < -10:
        flags.append("BSAP declining — reduced osteoblast activity")
    if trends.alp_delta_pct < -15:
        flags.append("ALP declining at week 6 — may indicate impaired bone metabolism")
    if trends.callus_delta_pct < 20:
        flags.append("Callus growth < 20% from Day 1 — risk of delayed union")
    if trends.p1np_delta_pct < 0:
        flags.append("P1NP trend negative — collagen synthesis may be impaired")
    return flags


def generate_recommendations(inference_result: dict, trends: BiomarkerTrends) -> list[str]:
    prob = inference_result.get("probability", 0.5)
    recs = []
    if prob < 0.45:
        recs.append("Consider early clinical intervention for non-union risk")
        recs.append("Repeat biomarker panel at week 9 if callus remains low")
    elif prob < 0.70:
        recs.append("Continue monitoring — schedule week-9 follow-up radiograph")
        recs.append("Nutritional assessment (calcium/vitamin D) recommended")
    else:
        recs.append("Healing trajectory appears normal — maintain current management")
        recs.append("Standard 12-week follow-up X-ray sufficient")
    return recs
