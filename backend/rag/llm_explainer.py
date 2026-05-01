from __future__ import annotations
import logging
import os

from backend.schemas.patient import PatientInput, BiomarkerTrends, SimilarCase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert orthopedic AI clinical assistant. Given patient biomarker data,
ML prediction results, similar historical cases, and relevant medical literature,
generate a concise clinical narrative (3-4 paragraphs) explaining the fracture
healing prediction. Use clinical but accessible language. Frame all statements as
predictions, not diagnoses. Do not recommend specific medications.
""".strip()


async def generate_clinical_explanation(
    patient: PatientInput,
    inference_result: dict,
    trends: BiomarkerTrends,
    similar_cases: list[SimilarCase],
    literature_snippets: list[str] | None = None,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — returning placeholder explanation.")
        return _fallback_explanation(patient, inference_result, trends)

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        similar_text = "\n".join(
            f"  • Case {c.case_id}: {c.age}yo {c.gender}, {c.fracture_location}, "
            f"callus_w6={c.callus_w6:.0f}, outcome={c.outcome} "
            f"(similarity={c.similarity_score:.0%})"
            for c in similar_cases
        ) or "  No similar cases found."

        lit_text = (
            "\n".join(f"  {s}" for s in (literature_snippets or []))
            or "  No literature context available."
        )

        user_message = f"""
Patient: {patient.age}yo {patient.gender.value}, fracture at {patient.fracture_location.value}

ML Prediction:
  Healing probability (Good): {round(inference_result['probability'] * 100)}%
  Predicted category: {inference_result['predicted_category']}
  Best model: {inference_result['best_model']}
  All model P(Good): {inference_result['good_probs']}

Biomarker trends (Day1 → Week3 → Week6):
  BSAP:   {trends.bsap_trend}  ({trends.bsap_delta_pct:+.1f}%)
  ALP:    {trends.alp_trend}   ({trends.alp_delta_pct:+.1f}%)
  P1NP:   {trends.p1np_trend}  ({trends.p1np_delta_pct:+.1f}%)
  Callus: {trends.callus_trend} ({trends.callus_delta_pct:+.1f}%)
  Summary: {trends.trend_summary}

Similar historical cases:
{similar_text}

Relevant medical literature:
{lit_text}

Please write a 3-4 paragraph clinical explanation of this patient's predicted fracture healing trajectory.
""".strip()

        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=700,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error(f"OpenAI call failed: {exc}")
        return _fallback_explanation(patient, inference_result, trends)


def _fallback_explanation(
    patient: PatientInput,
    inference_result: dict,
    trends: BiomarkerTrends,
) -> str:
    prob_pct = round(inference_result.get("probability", 0.5) * 100)
    cat = inference_result.get("predicted_category", "Moderate")
    return (
        f"Based on the provided biomarker and radiological data, this {patient.age}-year-old "
        f"{patient.gender.value} patient with a {patient.fracture_location.value} fracture "
        f"has a predicted healing probability of {prob_pct}%, classified as '{cat}' healing.\n\n"
        f"Biomarker trends show: {trends.trend_summary}\n\n"
        f"Clinical follow-up is recommended based on these results. "
        f"(AI explanation unavailable — OPENAI_API_KEY not configured.)"
    )
