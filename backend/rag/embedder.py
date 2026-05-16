from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.schemas.patient import PatientInput

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _patient_to_text(patient: PatientInput) -> str:
    return (
        f"Patient: {patient.age}yo {patient.gender.value}, "
        f"fracture at {patient.fracture_location.value}. "
        f"BSAP day1={patient.biomarkers_day1.bsap:.1f} week3={patient.biomarkers_week3.bsap:.1f}. "
        f"ALP day1={patient.biomarkers_day1.alp:.1f} week3={patient.biomarkers_week3.alp:.1f}. "
        f"P1NP day1={patient.biomarkers_day1.p1np:.1f} week3={patient.biomarkers_week3.p1np:.1f}. "
        f"Calcium week3={patient.minerals_week3.calcium:.1f}. "
        f"Callus day1={patient.callus_d1:.1f} week3={patient.callus_w3:.1f}."
    )


def embed_patient(patient: PatientInput) -> np.ndarray:
    return _get_model().encode(
        _patient_to_text(patient), normalize_embeddings=True
    )


def embed_text(text: str) -> np.ndarray:
    return _get_model().encode(text, normalize_embeddings=True)
