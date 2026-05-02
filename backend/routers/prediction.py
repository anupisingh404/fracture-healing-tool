from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from backend.schemas.patient import PatientInput, PredictionResult, BiomarkerTrends
from backend.ml.pipeline import (
    compute_trends, classify_category,
    extract_risk_flags, generate_recommendations,
)
from backend.ml.inference import run_inference
from backend.rag.retriever import retrieve_similar_cases
from backend.rag.llm_explainer import generate_clinical_explanation
from backend.rag.tavily_search import search_medical_literature

router = APIRouter()


def get_ml_pipeline(request: Request):
    """Load ML pipeline with error handling."""
    if not hasattr(request.app.state, 'ml_pipeline'):
        try:
            from backend.ml.pipeline import MLPipeline
            request.app.state.ml_pipeline = MLPipeline()
            request.app.state.ml_pipeline.load_or_train()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load ML models: {str(e)}")
    return request.app.state.ml_pipeline


def get_vector_store(request: Request):
    """Load vector store with error handling."""
    if not hasattr(request.app.state, 'vector_store'):
        try:
            from backend.rag.vector_store import VectorStore
            vs = VectorStore()
            vs.initialize()
            request.app.state.vector_store = vs
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
    return request.app.state.vector_store


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

    return PredictionResult(
        healing_probability=inference["probability"],
        healing_probability_pct=f"{round(inference['probability'] * 100)}% probability of successful healing",
        healing_category=classify_category(patient.callus_w6),
        model_used=inference["best_model"],
        confidence_scores=inference["good_probs"],
        biomarker_trends=trends,
        similar_cases=similar,
        clinical_explanation=explanation,
        risk_flags=extract_risk_flags(trends),
        recommendations=generate_recommendations(inference, trends),
    )


@router.post("/biomarker-trends", response_model=BiomarkerTrends)
def biomarker_trends(patient: PatientInput):
    return compute_trends(patient)


@router.get("/models")
def list_models(request: Request):
    ml = request.app.state.ml_pipeline
    return {
        "models": list(ml.models.keys()),
        "best_model": ml.best_model_name,
        "cv_scores": ml.cv_scores,
    }
