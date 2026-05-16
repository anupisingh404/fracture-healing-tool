from __future__ import annotations
import re
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class FractureLocation(str, Enum):
    FEMUR = "femur"
    TIBIA = "tibia"
    RADIUS = "radius"
    ULNA = "ulna"
    HUMERUS = "humerus"
    FIBULA = "fibula"
    PELVIS = "pelvis"
    VERTEBRA = "vertebra"


class HealingCategory(str, Enum):
    POOR = "Poor"
    MODERATE = "Moderate"
    GOOD = "Good"


class BiomarkerSet(BaseModel):
    bsap: float = Field(..., ge=0, description="Bone-Specific Alkaline Phosphatase (U/L)")
    alp: float = Field(..., ge=0, description="Alkaline Phosphatase (U/L)")
    p1np: float = Field(..., ge=0, description="N-terminal Propeptide of Type I Collagen (ng/mL)")


class MineralSet(BaseModel):
    calcium: float = Field(..., ge=0, description="Serum Calcium (mg/dL)")
    phosphorus: float = Field(..., ge=0, description="Serum Phosphorus (mg/dL)")


class PatientInput(BaseModel):
    patient_name: Optional[str] = Field(None, description="Patient full name")
    phone_no: Optional[str] = Field(None, description="Indian mobile number (+91 followed by 10 digits)")
    age: int = Field(..., ge=0, le=120)

    @field_validator("phone_no")
    @classmethod
    def validate_indian_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "":
            return v
        digits = re.sub(r"[\s\-()]", "", v)
        if not re.fullmatch(r"\+91[6-9]\d{9}", digits):
            raise ValueError("Phone number must be a valid Indian mobile number (e.g. +91 98765 43210)")
        return digits
    gender: Gender
    fracture_location: FractureLocation

    biomarkers_day1: BiomarkerSet
    biomarkers_week3: BiomarkerSet

    minerals_day1: MineralSet
    minerals_week3: MineralSet

    callus_d1: float = Field(..., ge=0, description="Callus measurement at Day 1 (mm²)")
    callus_w3: float = Field(..., ge=0, description="Callus measurement at Week 3 (mm²)")


class BiomarkerTrends(BaseModel):
    bsap_trend: list[float]
    alp_trend: list[float]
    p1np_trend: list[float]
    ca_trend: list[float]
    phos_trend: list[float]
    callus_trend: list[float]
    bsap_delta_pct: float
    alp_delta_pct: float
    p1np_delta_pct: float
    callus_delta_pct: float
    trend_summary: str


class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float
    patient_name: str = ""
    phone_no: str = ""
    age: int
    gender: str
    fracture_location: str
    callus_w3: float
    outcome: str
    summary: str


class PredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    case_id: str = ""
    healing_probability: float
    healing_probability_pct: str
    healing_category: HealingCategory
    model_used: str
    confidence_scores: dict[str, float]
    biomarker_trends: BiomarkerTrends
    similar_cases: list[SimilarCase]
    clinical_explanation: str
    risk_flags: list[str]
    recommendations: list[str]


class ConfirmOutcomeRequest(BaseModel):
    case_id: str
    actual_outcome: HealingCategory


class ConfirmOutcomeResponse(BaseModel):
    message: str
    case_id: str
    confirmed_count: int
    retrain_triggered: bool


class IngestCaseRequest(BaseModel):
    patient: PatientInput
    outcome: HealingCategory
    summary: Optional[str] = None
