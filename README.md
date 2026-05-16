# FractureAI — AI-Based Early Prediction Tool for Fracture Healing

An intelligent clinical web platform that predicts fracture healing outcomes using serum biomarkers (BSAP, ALP, P1NP), mineral markers (Calcium, Phosphorus), and radiological callus data measured at **Day 1** and **Week 3** post-fracture.

---

## Table of Contents

1. [Why This Tool Exists](#why-this-tool-exists)
2. [What It Does](#what-it-does)
3. [Architecture Overview](#architecture-overview)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [How It Works — Flow](#how-it-works--flow)
7. [Confirmed-Outcome Learning](#confirmed-outcome-learning)
8. [API Reference](#api-reference)
9. [MCP Integration (Claude Desktop)](#mcp-integration-claude-desktop)
10. [Dataset](#dataset)

---

## Why This Tool Exists

Fracture healing is currently monitored through X-rays and clinical symptoms — both of which detect problems **after** the optimal intervention window has closed.

| Current Problem | Impact |
|---|---|
| X-rays show healing changes late | Confirmation of union takes 6–12 weeks |
| No integrated biomarker system | BSAP, ALP, P1NP data collected but never used for prognosis |
| Non-union identified too late | Affects 5–10% of fractures, caught only at 3–6 months |
| No early prediction tools | Doctors lack actionable data at the critical Week 3 window |

**FractureAI solves this** by feeding routinely collected biomarker data through a 5-model ML ensemble, retrieving similar historical patient cases via RAG, and generating a structured clinical explanation using GPT-4o — all within seconds of entering patient data.

---

## What It Does

- **Predicts healing probability** (e.g. "82% probability of successful healing") and classifies the outcome as **Poor / Moderate / Good** based on Callus Week 3 thresholds
- **Runs 5 ML models in parallel** — Random Forest, XGBoost, Logistic Regression, SVM, Gradient Boosting — and shows all scores
- **Retrieves 3 similar historical patients** from a ChromaDB vector store using semantic similarity on biomarker summaries
- **Searches medical literature** via Tavily to find relevant research for the patient's fracture type and biomarker pattern
- **Generates a GPT-4o clinical narrative** grounded in ML results, biomarker trends, similar cases, and live literature
- **Outcome correction learning** — every prediction is saved immediately; if a clinician later confirms a different outcome, a corrected row is added and the model retrains instantly
- **No data loss if unconfirmed** — the predicted label is already in the training CSV, so the model learns from every patient regardless of whether the clinician confirms
- **Exposes all tools via MCP** (Model Context Protocol) so Claude Desktop can call them directly for agentic clinical workflows
- **Validates Indian phone numbers** — accepts `+91 XXXXX XXXXX` format in the patient form
- **Visualises biomarker trends** across Day 1 → Week 3 as interactive Chart.js line charts

### Healing Category Thresholds (Callus at Week 3)

| Category | Callus at Week 3 |
|---|---|
| Poor | < 40 mm² |
| Moderate | 40 – 72 mm² |
| Good | > 72 mm² |

---

## Architecture Overview

```
Browser (HTML/CSS/JS)
        │
        │  HTTP
        ▼
┌──────────────────────────────────────────────┐
│              FastAPI Backend                 │
│                                              │
│  POST /api/v1/prediction/predict  ──────────►│── ML Pipeline (5 models, .pkl)
│  POST /api/v1/prediction/confirm-outcome ───►│── Confirmed-outcome store + retrain
│  POST /api/v1/prediction/retrain  ──────────►│── Manual retrain from confirmed CSV
│  POST /api/v1/rag/similar-cases   ──────────►│── ChromaDB (grows with every patient)
│  GET  /api/v1/prediction/models   ──────────►│── Model scores & CV results
│  /mcp/call                        ──────────►│── MCP Tool Dispatcher
│  /                                ──────────►│── Static Frontend Files
└──────────────────────────────────────────────┘
        │                │
        ▼                ▼
   OpenAI GPT-4o    Tavily Search
  (clinical narrative) (medical literature)

        │  On predict:
        ▼
  pending_patients.json  ←── patient stored (keyed by case_id UUID)
  ChromaDB               ←── patient embedding stored for similarity search

        │  On confirm-outcome:
        ▼
  sample_patients.csv  ←── confirmed patient row appended (real label)

        │  Every 10 confirmed outcomes (background thread):
        ▼
  Models retrained → saved_models/*.pkl updated
```

### Component Breakdown

| Component | Technology | Purpose |
|---|---|---|
| Web Framework | FastAPI + Uvicorn | Async API server, static file serving |
| ML Models | scikit-learn, XGBoost | 5 classifiers for healing prediction |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | Local, free 384-dim semantic vectors |
| Vector Database | ChromaDB | Similar patient case retrieval |
| Web Search | Tavily API | Real-time medical literature retrieval |
| LLM | OpenAI GPT-4o | Clinical narrative generation |
| Protocol | MCP (Model Context Protocol) | Claude Desktop / agent tool integration |
| Frontend | HTML + CSS + JS + Chart.js | No bundler, no framework |

---

## Project Structure

```
fracture-healing-tool/
│
├── backend/
│   ├── main.py                    # FastAPI app entry point; sets ANONYMIZED_TELEMETRY
│   │                              #   before chromadb import to suppress telemetry errors
│   ├── config.py                  # Pydantic settings (reads .env into settings object)
│   │
│   ├── schemas/
│   │   └── patient.py             # All Pydantic models:
│   │                              #   PatientInput (patient_name, phone_no with +91 validation,
│   │                              #     age, gender, fracture_location, biomarkers Day1+Week3,
│   │                              #     minerals Day1+Week3, callus Day1+Week3)
│   │                              #   PredictionResult (includes case_id UUID)
│   │                              #   BiomarkerTrends, SimilarCase, HealingCategory
│   │                              #   ConfirmOutcomeRequest, ConfirmOutcomeResponse
│   │
│   ├── ml/
│   │   ├── pipeline.py            # featurize() → 24-dim feature vector, compute_trends(),
│   │   │                          #   classify_category() (callus_w3 thresholds),
│   │   │                          #   risk flags, recommendations,
│   │   │                          #   store_pending_patient(), confirm_pending_patient(),
│   │   │                          #   save_patient_to_csv(), retrain()
│   │   ├── trainer.py             # Trains all 5 models with StratifiedKFold CV, saves .pkl
│   │   ├── inference.py           # run_inference() → per-model probabilities + ensemble
│   │   └── saved_models/          # Persisted .pkl files (scaler + models + cv_scores +
│   │                              #   best_model_name)
│   │
│   ├── rag/
│   │   ├── embedder.py            # Loads MiniLM, embed_patient(), embed_text()
│   │   ├── vector_store.py        # ChromaDB client, initialize(), add_case() (stores
│   │   │                          #   patient_name + phone_no), query_cases(),
│   │   │                          #   seeds real_patients.csv on first run
│   │   ├── retriever.py           # retrieve_similar_cases() → list[SimilarCase]
│   │   ├── tavily_search.py       # search_medical_literature() — uses settings.tavily_api_key
│   │   └── llm_explainer.py       # GPT-4o clinical narrative — uses settings.openai_api_key,
│   │                              #   fallback if no key
│   │
│   ├── mcp_server/
│   │   ├── server.py              # MCP JSON-RPC 2.0 router (/mcp/manifest, /mcp/call)
│   │   └── tools.py               # 4 MCP tool handlers + dispatch()
│   │
│   ├── routers/
│   │   ├── prediction.py          # POST /predict (saves to CSV immediately + stores pending),
│   │   │                          #   POST /confirm-outcome (retrain only if outcome changed),
│   │   │                          #   POST /retrain (manual), POST /biomarker-trends,
│   │   │                          #   GET /models
│   │   ├── rag.py                 # POST /similar-cases, /ingest-case, GET /stats
│   │   └── mcp.py                 # Re-exports MCP server router
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── index.html                 # Landing page
│   ├── dashboard.html             # Doctor input form (Full Name, Indian Phone Number, Age,
│   │                              #   Gender default Female, Fracture Location, biomarkers
│   │                              #   Day1+Week3, minerals Day1+Week3, callus Day1+Week3)
│   │                              #   + results panel with Confirm Outcome card
│   ├── style.css                  # Design system: CSS variables, cards, gauge, charts
│   └── app.js                     # Form collection, fetch, Chart.js rendering, gauge SVG,
│                                  #   confirm-outcome handler (stores case_id, calls API)
│
├── data/
│   ├── real_patients.csv          # 30 real de-identified patients (used for ChromaDB seeding)
│   ├── sample_patients.csv        # ML training data — every prediction appended immediately;
│   │                              #   corrected rows added when clinician overrides outcome
│   ├── pending_patients.json      # Tracks predicted outcome per case_id so confirm-outcome
│   │                              #   can detect corrections; auto-cleaned on confirmation
│   └── generate_synthetic.py      # Script to regenerate base synthetic training data
│
├── chroma_db/                     # ChromaDB persistent store (grows with every prediction)
│
├── .env                           # All environment variables (API keys, paths, ML config)
└── README.md
```

---

## Setup

### Platform Compatibility

| Platform | Status | Notes |
|---|---|---|
| **Linux** (Ubuntu 20.04+, Debian, Fedora) | Fully supported | Primary development platform |
| **macOS** (12 Monterey+, Intel & Apple Silicon) | Fully supported | See Mac-specific notes below |
| **Windows** (10/11) | Supported via WSL2 | Native Windows not tested; use WSL2 |

### Prerequisites

- Python 3.11 or 3.12
- pip

> **macOS — install Python via Homebrew** (the system Python shipped with macOS is too old):
> ```bash
> brew install python@3.12
> ```
> On Apple Silicon (M1/M2/M3) the binary is at `/opt/homebrew/bin/python3`; on Intel at `/usr/local/bin/python3`.

### 1. Clone the repository

```bash
git clone git@github.com:anupisingh404/fracture-healing-tool.git
cd fracture-healing-tool
```

### 2. Create and activate a virtual environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (WSL2):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

> **macOS note:** If `pip install` fails on `chromadb` or `hnswlib` with a C++ compiler error, install the Xcode command-line tools first:
> ```bash
> xcode-select --install
> ```
> On Apple Silicon you may also need:
> ```bash
> brew install cmake
> ```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# ── External API Keys ─────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...          # Required for GPT-4o clinical explanations
TAVILY_API_KEY=tvly-...        # Required for medical literature search

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db          # Where ChromaDB stores its data
ANONYMIZED_TELEMETRY=false              # Suppresses ChromaDB posthog telemetry errors

# ── ML Pipeline ───────────────────────────────────────────────────────────────
MODEL_SAVE_DIR=./backend/ml/saved_models  # Where trained .pkl files are saved
BEST_ML_MODEL=XGBoost                     # Starting best model (updated after each retrain)
RETRAIN_EVERY_N=10                        # Auto-retrain after every N confirmed outcomes

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2   # Local embedding model for ChromaDB

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO                  # Options: DEBUG, INFO, WARNING, ERROR
```

**Variable reference:**

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | `""` | GPT-4o clinical explanation. Falls back to rule-based summary if not set |
| `TAVILY_API_KEY` | No | `""` | Medical literature search. Skipped gracefully if not set |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | ChromaDB persistent storage directory |
| `ANONYMIZED_TELEMETRY` | No | `false` | Set to `false` to suppress ChromaDB posthog errors |
| `MODEL_SAVE_DIR` | No | `./backend/ml/saved_models` | Directory for trained `.pkl` model files |
| `BEST_ML_MODEL` | No | `XGBoost` | Initial best model name (overridden after retraining) |
| `RETRAIN_EVERY_N` | No | `10` | Number of confirmed outcomes that triggers a background auto-retrain |
| `EMBED_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for patient embeddings |
| `LOG_LEVEL` | No | `INFO` | Python logging level |

> **Minimum setup:** only `OPENAI_API_KEY` and `TAVILY_API_KEY` need to be changed — all other variables have sensible defaults.

### 5. Generate synthetic training data (first time only)

```bash
python3 data/generate_synthetic.py
```

This writes `data/sample_patients.csv` (1000+ rows) used to train the ML models.

### 6. Train the ML models (first time only)

```bash
PYTHONPATH=. python3 -m backend.ml.trainer
```

This trains all 5 models with 5-fold cross-validation and saves `.pkl` files to `backend/ml/saved_models/`. Typical output:

```
RandomForest:       CV f1_macro = 0.9301
XGBoost:            CV f1_macro = 0.9463
LogisticRegression: CV f1_macro = 0.9572
SVM:                CV f1_macro = 0.9664
GradientBoosting:   CV f1_macro = 0.9451
Best model: SVM (0.9664)
```

> Steps 5 and 6 only need to be run once. On subsequent starts the server loads the saved `.pkl` files directly.

### 7. Start the server

**Linux / macOS:**
```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Windows (WSL2):** same command — run it inside the WSL2 terminal.

On first startup, ChromaDB is automatically seeded with all 30 real patients from `data/real_patients.csv` and 10 medical literature chunks (~30 seconds).

### 8. Open in browser

| URL | Page |
|---|---|
| `http://localhost:8000` | Landing page |
| `http://localhost:8000/dashboard` | Doctor dashboard |
| `http://localhost:8000/docs` | FastAPI auto-generated API docs |

---

## How It Works — Flow

### Full Request Flow (POST /api/v1/prediction/predict)

```
Doctor fills form on dashboard.html
  Fields: Full Name, Phone Number (+91 format), Age, Gender,
          Fracture Location, Biomarkers (BSAP/ALP/P1NP × Day1+Week3),
          Minerals (Ca/Phos × Day1+Week3), Callus (Day1+Week3)
        │
        │  POST /api/v1/prediction/predict  (JSON: PatientInput)
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Feature Engineering  (pipeline.py)             │
│                                                         │
│  PatientInput → 24-dimensional numpy feature vector     │
│                                                         │
│  Raw features (15):                                     │
│    age, gender, fracture_location                       │
│    bsap/alp/p1np × [day1, week3]                        │
│    calcium/phosphorus × [day1, week3]                   │
│    callus × [day1, week3]                               │
│                                                         │
│  Engineered features (9):                               │
│    bsap_delta, alp_delta, p1np_delta (day1→week3)       │
│    ca_delta, phos_delta                                 │
│    callus_delta (week3 - day1)                          │
│    bsap_alp_ratio_w3, ca_phos_product_w3                │
│    callus_growth_rate (mm²/day over 21 days)            │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: ML Inference  (inference.py)                   │
│                                                         │
│  feature_vector → StandardScaler → 5 classifiers        │
│  Each returns P(Poor), P(Moderate), P(Good)             │
│  Best model (highest CV score) → primary result         │
│                                                         │
│  Output:                                                │
│    healing_probability = P(Good) from best model        │
│    predicted_category  = argmax of best model           │
│    all_model_scores    = { model: P(Good), ... }        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Trend Analysis  (pipeline.py)                  │
│                                                         │
│  Per marker: trend list [d1, w3] + delta_pct            │
│  Rule-based narrative + risk flags + recommendations     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: RAG — Similar Case Retrieval  (retriever.py)   │
│                                                         │
│  Patient → MiniLM-L6-v2 → 384-dim vector                │
│  Cosine similarity search in ChromaDB                   │
│  Returns top-3 most similar cases with:                 │
│    patient_name, phone_no, age, gender,                 │
│    fracture_location, callus_w3, outcome, similarity%   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: Tavily Web Search  (tavily_search.py)          │
│                                                         │
│  Uses settings.tavily_api_key (loaded from .env)        │
│  Returns 3–5 medical literature snippets                │
│  Skipped gracefully if TAVILY_API_KEY not set           │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6: GPT-4o Clinical Explanation  (llm_explainer)   │
│                                                         │
│  Uses settings.openai_api_key (loaded from .env)        │
│  Prompt: patient data + ML scores + trends +            │
│          similar cases + literature snippets            │
│  Output: 3–4 paragraph clinical narrative               │
│  Fallback: rule-based summary if no OPENAI_API_KEY      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 7: Save & Track  (prediction.py router)           │
│                                                         │
│  sample_patients.csv: row appended immediately with     │
│    predicted label (model learns even if never          │
│    confirmed)                                           │
│  pending_patients.json: case_id → predicted_outcome     │
│    stored so confirm-outcome can detect corrections     │
│  ChromaDB: patient embedding stored for similarity      │
│                                                         │
│  Response includes case_id — shown in UI so clinician   │
│  can correct the outcome if the prediction was wrong    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Dashboard renders:
  • Animated SVG gauge (healing %)
  • Category badge (Good / Moderate / Poor)
  • All-model score bar chart
  • 2 Chart.js line charts (BSAP, ALP, P1NP, Callus — Day1→Week3)
  • Similar patients table (name, phone, similarity %)
  • GPT-4o clinical narrative
  • Risk flags + recommendations
  • Confirm Actual Outcome card (case_id + outcome dropdown)
```

### Startup Flow

```
uvicorn backend.main:app
        │
        ├── os.environ["ANONYMIZED_TELEMETRY"] = "false"   ← must be before chromadb import
        │
        ├── Load ML Pipeline
        │     • Check backend/ml/saved_models/ for .pkl files
        │     • If found → load scaler + models + best_model_name (fast, ~0.5s)
        │     • If not found → train on data/sample_patients.csv (~60s)
        │
        ├── Initialise Vector Store
        │     • Connect to ChromaDB at ./chroma_db/
        │     • If patient_cases empty → seed 30 real patients
        │     • If literature empty → seed 10 knowledge chunks
        │
        └── Server ready → http://0.0.0.0:8000
```

---

## Confirmed-Outcome Learning

Every prediction is saved immediately to the training CSV. If a clinician later confirms the real outcome and it **differs** from what was predicted, a corrected row is added and the model retrains immediately — so it learns from its own mistakes.

### How It Works

```
1. Doctor submits patient data
        │
        ▼
   predict endpoint runs ML + RAG + LLM
   ┌─────────────────────────────────────────────┐
   │ sample_patients.csv ← row saved immediately │
   │   with predicted label                      │
   │ pending_patients.json ← case_id mapped to   │
   │   predicted_outcome (for correction tracking)│
   │ ChromaDB ← embedding stored                 │
   └─────────────────────────────────────────────┘
        │
        ▼
2. Prediction shown on dashboard
   "Confirm Actual Outcome" card shows case_id + outcome dropdown
   (pre-filled with the predicted outcome)
        │
        ├─── Clinician never confirms
        │      → No problem. Predicted label already in CSV.
        │        Model already learned from this patient.
        │
        └─── Clinician confirms outcome
                │
                ├── actual == predicted
                │     → "Matches prediction. No retraining needed."
                │       Entry removed from pending. CSV already correct.
                │
                └── actual != predicted
                      → Corrected row added to sample_patients.csv
                        Model retrained immediately (background thread)
                        "Outcome corrected (Poor → Good). Retraining triggered."
```

### Confirm via API

```bash
curl -X POST http://localhost:8000/api/v1/prediction/confirm-outcome \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "392ed4d0-7c0e-4420-b572-83095691c58a",
    "actual_outcome": "Good"
  }'
```

**Response when outcome matches prediction:**
```json
{
  "message": "Outcome confirmed as 'Good' — matches prediction. No retraining needed.",
  "case_id": "392ed4d0-7c0e-4420-b572-83095691c58a",
  "confirmed_count": 0,
  "retrain_triggered": false
}
```

**Response when outcome differs from prediction:**
```json
{
  "message": "Outcome corrected (Poor → Good). Corrected row saved and model retraining triggered.",
  "case_id": "392ed4d0-7c0e-4420-b572-83095691c58a",
  "confirmed_count": 0,
  "retrain_triggered": true
}
```

### Behaviour Summary

| Scenario | CSV | Retrain |
|---|---|---|
| Clinician never confirms | Predicted label saved on predict | No |
| Confirms — same as predicted | Already correct, no change | No |
| Confirms — different from predicted | Corrected row appended | Yes — immediately |

### Manual Retrain

Trigger a retrain at any time:

```bash
curl -X POST http://localhost:8000/api/v1/prediction/retrain
```

Response:
```json
{
  "status": "retrained",
  "training_rows": 1042,
  "cv_scores": { "LogisticRegression": 0.8817, "RandomForest": 0.8583, ... },
  "best_model": "LogisticRegression"
}
```

---

## API Reference

### Prediction

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/prediction/predict` | Full pipeline — returns `PredictionResult` with `case_id` |
| POST | `/api/v1/prediction/confirm-outcome` | If outcome differs from prediction: save corrected row + retrain immediately |
| POST | `/api/v1/prediction/retrain` | Manually retrain all ML models from current CSV |
| POST | `/api/v1/prediction/biomarker-trends` | Trend analysis only (no LLM, fast) |
| GET | `/api/v1/prediction/models` | Lists loaded models and CV scores |

**Sample request body for `/predict`:**

```json
{
  "patient_name": "Anupi Singh",
  "phone_no": "+91 98765 43210",
  "age": 26,
  "gender": "female",
  "fracture_location": "tibia",
  "biomarkers_day1":  { "bsap": 28.0, "alp": 75.0,  "p1np": 50.0 },
  "biomarkers_week3": { "bsap": 36.0, "alp": 90.0,  "p1np": 65.0 },
  "minerals_day1":    { "calcium": 9.5,  "phosphorus": 3.5 },
  "minerals_week3":   { "calcium": 9.3,  "phosphorus": 3.4 },
  "callus_d1": 20.0,
  "callus_w3": 80.0
}
```

> `patient_name` and `phone_no` are optional — predictions work without them.
> Phone numbers must be valid Indian mobile numbers: `+91` followed by a digit 6–9 then 9 more digits.

Valid `fracture_location` values: `femur`, `tibia`, `radius`, `ulna`, `humerus`, `fibula`, `pelvis`, `vertebra`

**Sample `PredictionResult` response (abbreviated):**

```json
{
  "case_id": "392ed4d0-7c0e-4420-b572-83095691c58a",
  "healing_probability": 0.7251,
  "healing_probability_pct": "73% probability of successful healing",
  "healing_category": "Good",
  "model_used": "LogisticRegression",
  "confidence_scores": { "LogisticRegression": 0.73, "RandomForest": 0.68, ... },
  "biomarker_trends": { "bsap_trend": [28.0, 36.0], "bsap_delta_pct": 28.57, ... },
  "similar_cases": [ { "patient_name": "...", "callus_w3": 82.0, "outcome": "Good", ... } ],
  "clinical_explanation": "...",
  "risk_flags": [],
  "recommendations": ["Healing trajectory appears normal — maintain current management.", ...]
}
```

### RAG

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/rag/similar-cases` | Returns top-3 similar patients from ChromaDB |
| POST | `/api/v1/rag/ingest-case` | Manually adds a case to the vector store |
| GET | `/api/v1/rag/stats` | Returns `{ total_cases, literature_chunks }` |

### MCP

| Method | Endpoint | Description |
|---|---|---|
| GET | `/mcp/manifest` | Returns full tool manifest JSON |
| POST | `/mcp/call` | Calls a tool by name (JSON-RPC 2.0) |

**MCP tools available:**

| Tool | Description |
|---|---|
| `predict_fracture_healing` | ML prediction from patient biomarkers |
| `analyze_biomarker_trends` | % trend changes across time points |
| `get_similar_cases` | ChromaDB similarity retrieval |
| `explain_prediction` | GPT-4o + RAG + Tavily explanation |

---

## MCP Integration (Claude Desktop)

Register this server in Claude Desktop's `claude_desktop_config.json`.

**Config file location by platform:**

| Platform | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "fracture-healing": {
      "url": "http://localhost:8000/mcp/call",
      "type": "http"
    }
  }
}
```

Claude can then call tools directly:

> *"Predict healing for a 26-year-old female with tibia fracture. BSAP day1=28, week3=36. ALP day1=75, week3=90. Callus day1=20, week3=80."*

Claude will invoke `predict_fracture_healing`, chain it with `explain_prediction`, and return a structured clinical report without any additional code.

---

## Dataset

### Real Patients (`data/real_patients.csv`)

- **30 patients** from actual clinical records
- Fracture types: Tibia (12), Femur (8), Humerus (5), Ulna (3), Radius (2)
- Gender: Male (21), Female (9)
- Used exclusively for initial ChromaDB seeding (RAG)
- Contains: `patient_name`, `phone_no`, `age`, `gender`, `fracture_location`, full biomarker panel, `healing_category`

### Training Data (`data/sample_patients.csv`)

- **Starts with 1000+ synthetic patients** generated with realistic statistical distributions
- **Grows with confirmed outcomes** — every clinician-confirmed case appends a row with a real label
- Distribution (synthetic base): Good (400), Moderate (350), Poor (250)
- Regenerate base synthetic data: `python3 data/generate_synthetic.py`

### Pending Store (`data/pending_patients.json`)

- Temporary JSON file keyed by UUID case_id
- Holds patient data between prediction and outcome confirmation
- Entry is removed automatically when the clinician confirms the outcome
- Safe to inspect or clear manually if needed

### Why Two CSV Datasets?

The real dataset has only 30 patients — too few for reliable 5-fold cross-validation. Synthetic data provides the base for initial training. As clinicians confirm real patient outcomes, those rows accumulate in `sample_patients.csv`, and the models progressively retrain on real-world distributions.

### Data Flow Summary

```
First run:
  data/real_patients.csv (30)  ──► ChromaDB (seeded once, for RAG)
  data/sample_patients.csv     ──► ML models trained (synthetic base)

On every prediction:
  New patient + predicted label ──► sample_patients.csv (immediate)
  case_id → predicted_outcome   ──► pending_patients.json (for correction tracking)
  New patient embedding         ──► ChromaDB (for similarity search)

On clinician confirmation — same outcome:
  Entry removed from pending_patients.json
  (CSV already correct — no further action)

On clinician confirmation — different outcome:
  Corrected patient + real label ──► sample_patients.csv (new row)
  Entry removed from pending_patients.json
  Models retrained immediately from full CSV (background thread)
```
