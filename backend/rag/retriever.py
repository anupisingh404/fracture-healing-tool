from __future__ import annotations
from backend.schemas.patient import PatientInput, SimilarCase
from backend.rag.vector_store import VectorStore


def retrieve_similar_cases(
    store: VectorStore, patient: PatientInput, k: int = 3
) -> list[SimilarCase]:
    raw = store.query_cases(patient, k=k)
    return [
        SimilarCase(
            case_id=r["case_id"],
            similarity_score=r["similarity_score"],
            patient_name=r.get("patient_name", ""),
            phone_no=r.get("phone_no", ""),
            age=int(r["age"]),
            gender=r["gender"],
            fracture_location=r["fracture_location"],
            callus_w6=float(r["callus_w6"]),
            outcome=r["outcome"],
            summary=r["summary"],
        )
        for r in raw
    ]
