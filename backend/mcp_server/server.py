from __future__ import annotations
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Any

from backend.mcp_server.tools import dispatch

router = APIRouter()

PATIENT_INPUT_SCHEMA = {
    "type": "object",
    "required": [
        "age", "gender", "fracture_location",
        "biomarkers_day1", "biomarkers_week3",
        "minerals_day1", "minerals_week3",
        "callus_d1", "callus_w3",
    ],
    "properties": {
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "gender": {"type": "string", "enum": ["male", "female", "other"]},
        "fracture_location": {
            "type": "string",
            "enum": ["femur", "tibia", "radius", "humerus", "fibula", "pelvis", "vertebra"],
        },
        "biomarkers_day1":  {"$ref": "#/definitions/BiomarkerSet"},
        "biomarkers_week3": {"$ref": "#/definitions/BiomarkerSet"},
        "minerals_day1":    {"$ref": "#/definitions/MineralSet"},
        "minerals_week3":   {"$ref": "#/definitions/MineralSet"},
        "callus_d1":  {"type": "number", "minimum": 0},
        "callus_w3":  {"type": "number", "minimum": 0},
    },
    "definitions": {
        "BiomarkerSet": {
            "type": "object",
            "properties": {
                "bsap":  {"type": "number", "minimum": 0},
                "alp":   {"type": "number", "minimum": 0},
                "p1np":  {"type": "number", "minimum": 0},
            },
        },
        "MineralSet": {
            "type": "object",
            "properties": {
                "calcium":    {"type": "number", "minimum": 0},
                "phosphorus": {"type": "number", "minimum": 0},
            },
        },
    },
}

MCP_TOOLS = [
    {
        "name": "predict_fracture_healing",
        "description": "Predict fracture healing probability and category from patient biomarkers",
        "inputSchema": PATIENT_INPUT_SCHEMA,
    },
    {
        "name": "analyze_biomarker_trends",
        "description": "Analyze trend changes in BSAP, ALP, P1NP, and callus across Day1/Week3",
        "inputSchema": PATIENT_INPUT_SCHEMA,
    },
    {
        "name": "get_similar_cases",
        "description": "Retrieve similar historical patient cases from the vector store",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient": PATIENT_INPUT_SCHEMA,
                "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
            },
        },
    },
    {
        "name": "explain_prediction",
        "description": "Generate a clinical narrative explanation using GPT-4o, RAG, and Tavily literature search",
        "inputSchema": PATIENT_INPUT_SCHEMA,
    },
]


class MCPCallRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Any = None
    method: str
    params: dict = {}


class MCPCallResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Any = None
    result: Any = None
    error: Any = None


@router.get("/manifest")
def mcp_manifest():
    return {
        "schema_version": "v1",
        "name": "fracture-healing-tool",
        "description": "MCP server for AI-based fracture healing prediction",
        "tools": MCP_TOOLS,
    }


@router.get("/tools")
def mcp_tools_list():
    return {"tools": MCP_TOOLS}


@router.post("/call", response_model=MCPCallResponse)
async def mcp_call(req: MCPCallRequest, request: Request):
    try:
        result = await dispatch(req.method, req.params, request.app.state)
        return MCPCallResponse(id=req.id, result=result)
    except ValueError as exc:
        return MCPCallResponse(
            id=req.id,
            error={"code": -32601, "message": str(exc)},
        )
    except Exception as exc:
        return MCPCallResponse(
            id=req.id,
            error={"code": -32603, "message": f"Internal error: {exc}"},
        )
