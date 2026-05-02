from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager

# Must be set before chromadb is imported to suppress posthog telemetry errors
from backend.config import settings
os.environ["ANONYMIZED_TELEMETRY"] = settings.anonymized_telemetry

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.ml.pipeline import MLPipeline
from backend.rag.vector_store import VectorStore
from backend.routers import prediction, rag
from backend.mcp_server.server import router as mcp_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Fracture Healing Prediction API",
    version="1.0.0",
    description="AI-Based Early Prediction Tool for Fracture Healing",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["Prediction"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(mcp_router, prefix="/mcp", tags=["MCP"])

# Serve frontend static files
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/dashboard", include_in_schema=False)
    def serve_dashboard():
        return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

    @app.get("/{filename}", include_in_schema=False)
    def serve_static(filename: str):
        path = os.path.join(FRONTEND_DIR, filename)
        if os.path.isfile(path):
            return FileResponse(path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
