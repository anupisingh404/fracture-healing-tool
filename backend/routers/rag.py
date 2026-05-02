from __future__ import annotations
import uuid
from fastapi import APIRouter, Request, HTTPException
from backend.schemas.patient import IngestCaseRequest, SimilarCase
from backend.rag.retriever import retrieve_similar_cases

router = APIRouter()


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


@router.post("/similar-cases")
def similar_cases(patient_data: IngestCaseRequest, request: Request):
    vs = get_vector_store(request)
    cases = retrieve_similar_cases(vs, patient_data.patient, k=3)
    return {"cases": [c.model_dump() for c in cases]}


@router.post("/ingest-case")
def ingest_case(body: IngestCaseRequest, request: Request):
    vs = get_vector_store(request)
    cid = str(uuid.uuid4())
    vs.add_case(body.patient, body.outcome, body.summary, case_id=cid)
    return {"doc_id": cid, "status": "ingested"}


@router.get("/stats")
def rag_stats(request: Request):
    vs = get_vector_store(request)
    return vs.stats()
