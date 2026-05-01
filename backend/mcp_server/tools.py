from __future__ import annotations
from backend.schemas.patient import PatientInput, HealingCategory
from backend.ml.pipeline import compute_trends, classify_category, extract_risk_flags, generate_recommendations
from backend.ml.inference import run_inference
from backend.rag.retriever import retrieve_similar_cases
from backend.rag.llm_explainer import generate_clinical_explanation
from backend.rag.tavily_search import search_medical_literature


async def dispatch(tool_name: str, args: dict, app_state) -> dict:
    handlers = {
        "predict_fracture_healing": _predict,
        "analyze_biomarker_trends": _trends,
        "get_similar_cases": _similar,
        "explain_prediction": _explain,
    }
    if tool_name not in handlers:
        raise ValueError(f"Unknown MCP tool: {tool_name}")
    return await handlers[tool_name](args, app_state)


async def _predict(args: dict, state) -> dict:
    patient = PatientInput(**args)
    features, _ = state.ml_pipeline.featurize(patient)
    result = run_inference(state.ml_pipeline, features)
    category = classify_category(patient.callus_w6)
    return {
        "healing_probability_pct": f"{round(result['probability'] * 100)}%",
        "healing_category": category.value,
        "predicted_category": result["predicted_category"],
        "model_used": result["best_model"],
        "all_model_scores": result["good_probs"],
        "ensemble_probability_pct": f"{round(result['ensemble_probability'] * 100)}%",
    }


async def _trends(args: dict, state) -> dict:
    patient = PatientInput(**args)
    trends = compute_trends(patient)
    flags = extract_risk_flags(trends)
    return {**trends.model_dump(), "risk_flags": flags}


async def _similar(args: dict, state) -> dict:
    patient_data = args.get("patient", args)
    k = args.get("k", 3)
    patient = PatientInput(**patient_data)
    cases = retrieve_similar_cases(state.vector_store, patient, k=k)
    return {"similar_cases": [c.model_dump() for c in cases]}


async def _explain(args: dict, state) -> dict:
    patient = PatientInput(**args)
    features, _ = state.ml_pipeline.featurize(patient)
    inference = run_inference(state.ml_pipeline, features)
    trends = compute_trends(patient)
    similar = retrieve_similar_cases(state.vector_store, patient, k=3)
    lit = search_medical_literature(
        patient.fracture_location.value,
        inference["predicted_category"],
        trends.trend_summary,
    )
    explanation = await generate_clinical_explanation(patient, inference, trends, similar, lit)
    return {"clinical_explanation": explanation}
