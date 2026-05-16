from __future__ import annotations
import asyncio
import logging
import os
import pandas as pd

from fastapi import APIRouter, Request, HTTPException
from backend.schemas.patient import (
    PatientInput, PredictionResult, BiomarkerTrends,
    ConfirmOutcomeRequest, ConfirmOutcomeResponse, HealingCategory,
)
from backend.ml.pipeline import (
    compute_trends, classify_category,
    extract_risk_flags, generate_recommendations, DATA_PATH,
    store_pending_patient, confirm_pending_patient,
)
from backend.ml.inference import run_inference
from backend.rag.retriever import retrieve_similar_cases
from backend.rag.llm_explainer import generate_clinical_explanation
from backend.rag.tavily_search import search_medical_literature
from backend.config import settings

logger = logging.getLogger(__name__)

RETRAIN_EVERY_N = settings.retrain_every_n

router = APIRouter()


def get_ml_pipeline(request: Request):
    if not hasattr(request.app.state, 'ml_pipeline'):
        try:
            from backend.ml.pipeline import MLPipeline
            request.app.state.ml_pipeline = MLPipeline()
            request.app.state.ml_pipeline.load_or_train()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load ML models: {str(e)}")
    return request.app.state.ml_pipeline


def get_vector_store(request: Request):
    if not hasattr(request.app.state, 'vector_store'):
        try:
            from backend.rag.vector_store import VectorStore
            vs = VectorStore()
            vs.initialize()
            request.app.state.vector_store = vs
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
    return request.app.state.vector_store


def _new_patient_count(request: Request) -> int:
    if not hasattr(request.app.state, 'new_patient_count'):
        request.app.state.new_patient_count = 0
    return request.app.state.new_patient_count


def _confirmed_count(request: Request) -> int:
    if not hasattr(request.app.state, 'confirmed_count'):
        request.app.state.confirmed_count = 0
    return request.app.state.confirmed_count


async def _retrain_in_background(ml, total_rows: int):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ml.retrain)
    logger.info(f"Auto-retrain complete - trained on {total_rows} patients.")


@router.post("/predict", response_model=PredictionResult)
async def predict(patient: PatientInput, request: Request):
    ml = get_ml_pipeline(request)
    vs = get_vector_store(request)

    features, _ = ml.featurize(patient)
    inference = run_inference(ml, features)
    trends = compute_trends(patient)
    similar = retrieve_similar_cases(vs, patient, k=3)
    lit = search_medical_literature(
        patient.fracture_location.value,
        inference["predicted_category"],
        trends.trend_summary,
    )
    explanation = await generate_clinical_explanation(patient, inference, trends, similar, lit)
    healing_cat = classify_category(patient.callus_w3)

    # Store patient as pending — outcome will be confirmed by the clinician
    case_id = ""
    try:
        case_id = store_pending_patient(patient, healing_cat)
    except Exception as e:
        logger.warning(f"Failed to store pending patient: {e}")

    # Also add to ChromaDB for immediate similarity search use
    try:
        vs.add_case(patient, healing_cat)
    except Exception as e:
        logger.warning(f"Failed to save patient to ChromaDB: {e}")

    return PredictionResult(
        case_id=case_id,
        healing_probability=inference["probability"],
        healing_probability_pct=f"{round(inference['probability'] * 100)}% probability of successful healing",
        healing_category=healing_cat,
        model_used=inference["best_model"],
        confidence_scores=inference["good_probs"],
        biomarker_trends=trends,
        similar_cases=similar,
        clinical_explanation=explanation,
        risk_flags=extract_risk_flags(trends),
        recommendations=generate_recommendations(inference, trends),
    )


@router.post("/confirm-outcome", response_model=ConfirmOutcomeResponse)
async def confirm_outcome(req: ConfirmOutcomeRequest, request: Request):
    ml = get_ml_pipeline(request)

    patient_data = confirm_pending_patient(req.case_id, req.actual_outcome)
    if patient_data is None:
        raise HTTPException(status_code=404, detail=f"Case ID '{req.case_id}' not found or already confirmed.")

    # Reconstruct PatientInput and save with the real confirmed label
    try:
        patient = PatientInput(**patient_data)
        ml.save_patient_to_csv(patient, req.actual_outcome)
    except Exception as e:
        logger.error(f"Failed to save confirmed patient to CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to save confirmed outcome.")

    # Track and possibly trigger retrain
    request.app.state.confirmed_count = _confirmed_count(request) + 1
    confirmed = request.app.state.confirmed_count
    retrain_triggered = False

    if confirmed % RETRAIN_EVERY_N == 0:
        try:
            total_rows = len(pd.read_csv(os.path.abspath(DATA_PATH)))
            logger.info(f"Auto-retrain triggered after {confirmed} confirmed outcomes ({total_rows} total rows).")
            asyncio.create_task(_retrain_in_background(ml, total_rows))
            retrain_triggered = True
        except Exception as e:
            logger.warning(f"Failed to schedule retrain: {e}")

    return ConfirmOutcomeResponse(
        message=f"Outcome '{req.actual_outcome.value}' confirmed and saved for case {req.case_id}.",
        case_id=req.case_id,
        confirmed_count=confirmed,
        retrain_triggered=retrain_triggered,
    )


@router.post("/biomarker-trends", response_model=BiomarkerTrends)
def biomarker_trends(patient: PatientInput):
    return compute_trends(patient)


@router.get("/models")
def list_models(request: Request):
    ml = get_ml_pipeline(request)
    return {
        "models": list(ml.models.keys()),
        "best_model": ml.best_model_name,
        "cv_scores": ml.cv_scores,
    }


@router.post("/retrain")
def retrain_models(request: Request):
    ml = get_ml_pipeline(request)
    row_count = len(pd.read_csv(os.path.abspath(DATA_PATH)))
    cv_scores = ml.retrain()
    request.app.state.new_patient_count = 0
    request.app.state.confirmed_count = 0
    return {
        "status": "retrained",
        "training_rows": row_count,
        "cv_scores": cv_scores,
        "best_model": ml.best_model_name,
    }
