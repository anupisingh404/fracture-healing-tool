from __future__ import annotations
import os
import uuid
import logging
import pandas as pd
import chromadb
from chromadb.config import Settings

from backend.schemas.patient import PatientInput, HealingCategory
from backend.rag.embedder import embed_patient, embed_text

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
REAL_CSV = os.path.join(os.path.dirname(__file__), "../../data/real_patients.csv")


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.cases_col = None
        self.literature_col = None

    def initialize(self):
        self.cases_col = self.client.get_or_create_collection(
            name="patient_cases",
            metadata={"hnsw:space": "cosine"},
        )
        self.literature_col = self.client.get_or_create_collection(
            name="medical_literature",
            metadata={"hnsw:space": "cosine"},
        )
        if self.cases_col.count() == 0:
            self._seed_cases()
        if self.literature_col.count() == 0:
            self._seed_literature()
        logger.info(
            f"VectorStore ready -cases: {self.cases_col.count()}, "
            f"literature: {self.literature_col.count()}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_case(
        self,
        patient: PatientInput,
        outcome: HealingCategory,
        summary: str | None = None,
        case_id: str | None = None,
    ) -> str:
        cid = case_id or str(uuid.uuid4())
        name = getattr(patient, "patient_name", "") or ""
        phone = getattr(patient, "phone_no", "") or ""
        if summary is None:
            summary = (
                f"Patient: {name} (Phone: {phone}). "
                f"{patient.age}yo {patient.gender.value} with "
                f"{patient.fracture_location.value} fracture. "
                f"Callus week6={patient.callus_w6:.0f}. Outcome: {outcome.value}."
            )
        embedding = embed_patient(patient).tolist()
        self.cases_col.add(
            ids=[cid],
            embeddings=[embedding],
            metadatas=[{
                "age": patient.age,
                "gender": patient.gender.value,
                "fracture_location": patient.fracture_location.value,
                "callus_w6": patient.callus_w6,
                "outcome": outcome.value,
                "patient_name": name,
                "phone_no": phone,
                "summary": summary,
            }],
            documents=[summary],
        )
        return cid

    def query_cases(self, patient: PatientInput, k: int = 3) -> list[dict]:
        embedding = embed_patient(patient).tolist()
        results = self.cases_col.query(
            query_embeddings=[embedding],
            n_results=min(k, self.cases_col.count()),
            include=["metadatas", "distances", "documents"],
        )
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            out.append({
                **meta,
                "case_id": doc_id,
                "similarity_score": round(1 - results["distances"][0][i], 4),
            })
        return out

    def query_literature(self, query_text: str, k: int = 3) -> list[str]:
        embedding = embed_text(query_text).tolist()
        count = self.literature_col.count()
        if count == 0:
            return []
        results = self.literature_col.query(
            query_embeddings=[embedding],
            n_results=min(k, count),
            include=["documents"],
        )
        return results["documents"][0]

    def stats(self) -> dict:
        return {
            "total_cases": self.cases_col.count(),
            "literature_chunks": self.literature_col.count(),
        }

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _seed_cases(self):
        if not os.path.exists(REAL_CSV):
            logger.warning("real_patients.csv not found -skipping case seeding.")
            return
        df = pd.read_csv(REAL_CSV)  # all 30 real patients
        logger.info(f"Seeding {len(df)} real patient cases into ChromaDB …")
        for _, row in df.iterrows():
            outcome_str = row["healing_category"]
            outcome = HealingCategory(outcome_str)
            name = str(row.get("patient_name", "") or "").strip()
            raw_phone = row.get("phone_no", "")
            try:
                import math
                phone = str(int(float(raw_phone))) if raw_phone and not math.isnan(float(raw_phone)) else ""
            except (ValueError, TypeError):
                phone = str(raw_phone).strip() if raw_phone else ""
            summary = (
                f"Patient: {name} (Phone: {phone}). "
                f"{int(row['age'])}yo {row['gender']} with {row['fracture_location']} fracture. "
                f"BSAP day1={row['bsap_d1']:.1f}→week6={row['bsap_w6']:.1f}. "
                f"ALP day1={row['alp_d1']:.1f}→week6={row['alp_w6']:.1f}. "
                f"Callus week6={row['callus_w6']:.0f}. Outcome: {outcome_str}."
            )
            self.cases_col.add(
                ids=[row["patient_id"]],
                embeddings=[embed_text(summary).tolist()],
                metadatas=[{
                    "age": int(row["age"]),
                    "gender": row["gender"],
                    "fracture_location": row["fracture_location"],
                    "callus_w6": float(row["callus_w6"]),
                    "outcome": outcome_str,
                    "patient_name": str(name),
                    "phone_no": str(phone),
                    "summary": summary,
                }],
                documents=[summary],
            )

    def _seed_literature(self):
        chunks = [
            "BSAP (bone-specific alkaline phosphatase) is a sensitive marker of osteoblast activity. Rising BSAP levels in the first 6 weeks after fracture correlate with successful callus formation and good healing outcomes.",
            "ALP (alkaline phosphatase) elevation following fracture reflects bone remodelling. Studies show ALP peaks at 2–3 weeks post-fracture and sustained elevation beyond week 6 is associated with delayed union.",
            "P1NP (N-terminal propeptide of type 1 procollagen) is a collagen synthesis marker. High P1NP at week 3 predicts greater callus volume at week 6 in long-bone fractures.",
            "Radiological callus measurement at week 6 is the most reliable early predictor of fracture union. Callus area > 180 mm² at 6 weeks has been associated with successful union in femoral and tibial fractures.",
            "Calcium and phosphorus homeostasis is critical for mineralisation of the fracture callus. Hypocalcaemia in the early post-fracture period is associated with impaired callus mineralisation and delayed union.",
            "Machine learning models trained on biomarker time-series data have achieved 85–92% accuracy in predicting fracture healing outcomes at 6 weeks, outperforming single-timepoint radiological assessment alone.",
            "Non-union affects approximately 5–10% of all fractures. Risk factors include advanced age, smoking, diabetes, NSAID use, poor nutrition, and inadequate immobilisation.",
            "XGBoost and Random Forest classifiers have demonstrated superior performance on small clinical datasets (n < 100) compared to deep learning approaches due to their ability to handle high-dimensional tabular data with limited samples.",
            "Fracture healing progresses through four overlapping phases: haematoma formation (day 1–5), soft callus formation (week 1–3), hard callus formation (week 3–12), and bone remodelling (months to years).",
            "Early biomarker assessment at day 1 and week 3 allows identification of patients at risk for delayed union up to 3–4 weeks earlier than standard radiological follow-up, enabling timely intervention.",
        ]
        ids = [f"lit_{i}" for i in range(len(chunks))]
        embeddings = [embed_text(c).tolist() for c in chunks]
        self.literature_col.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": "medical_literature"} for _ in chunks],
        )
        logger.info(f"Seeded {len(chunks)} literature chunks.")
