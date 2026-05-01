from __future__ import annotations
import uuid
from fastapi import APIRouter, Request
from backend.schemas.patient import IngestCaseRequest, SimilarCase
from backend.rag.retriever import retrieve_similar_cases

router = APIRouter()


@router.post("/similar-cases")
def similar_cases(patient_data: IngestCaseRequest, request: Request):
    vs = request.app.state.vector_store
    cases = retrieve_similar_cases(vs, patient_data.patient, k=3)
    return {"cases": [c.model_dump() for c in cases]}


@router.post("/ingest-case")
def ingest_case(body: IngestCaseRequest, request: Request):
    vs = request.app.state.vector_store
    cid = str(uuid.uuid4())
    vs.add_case(body.patient, body.outcome, body.summary, case_id=cid)
    return {"doc_id": cid, "status": "ingested"}


@router.get("/stats")
def rag_stats(request: Request):
    return request.app.state.vector_store.stats()
