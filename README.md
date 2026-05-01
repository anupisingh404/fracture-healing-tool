# FractureAI — AI-Based Early Prediction Tool for Fracture Healing

An intelligent clinical web platform that predicts fracture healing outcomes using serum biomarkers (BSAP, ALP, P1NP), mineral markers (Calcium, Phosphorus), and radiological callus data measured at Day 1, Week 3, and Week 6 post-fracture.

---

## Table of Contents

1. [Why This Tool Exists](#why-this-tool-exists)
2. [What It Does](#what-it-does)
3. [Architecture Overview](#architecture-overview)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [How It Works — Flow](#how-it-works--flow)
7. [API Reference](#api-reference)
8. [MCP Integration (Claude Desktop)](#mcp-integration-claude-desktop)
9. [Dataset](#dataset)

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

- **Predicts healing probability** (e.g. "82% probability of successful healing") and classifies the outcome as **Poor / Moderate / Good** based on Callus_w6 thresholds
- **Runs 5 ML models in parallel** — Random Forest, XGBoost, Logistic Regression, SVM, Gradient Boosting — and shows all scores
- **Retrieves 3 similar historical patients** from a ChromaDB vector store using semantic similarity on biomarker summaries
- **Searches medical literature** via Tavily to find relevant research for the patient's fracture type and biomarker pattern
- **Generates a GPT-4o clinical narrative** grounded in ML results, biomarker trends, similar cases, and live literature
- **Exposes all tools via MCP** (Model Context Protocol) so Claude Desktop can call them directly for agentic clinical workflows
- **Visualises biomarker trends** across Day 1 → Week 3 → Week 6 as interactive Chart.js line charts

### Healing Category Thresholds (Callus_w6)

| Category | Callus at Week 6 |
|---|---|
| Poor | < 100 mm² |
| Moderate | 100 – 180 mm² |
| Good | > 180 mm² |

---

## Architecture Overview

```
Browser (HTML/CSS/JS)
        │
        │  HTTP
        ▼
┌─────────────────────────────────────┐
│           FastAPI Backend           │
│                                     │
│  /api/v1/prediction/predict  ──────►│── ML Pipeline (5 models, .pkl)
│  /api/v1/rag/similar-cases   ──────►│── ChromaDB (30 real patients)
│  /api/v1/rag/stats                  │
│  /mcp/call                   ──────►│── MCP Tool Dispatcher
│  /                           ──────►│── Static Frontend Files
└─────────────────────────────────────┘
        │                │
        │                │
        ▼                ▼
   OpenAI GPT-4o    Tavily Search
  (clinical narrative) (medical literature)
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
│   ├── main.py                    # FastAPI app entry point, lifespan startup
│   ├── config.py                  # Pydantic settings (reads .env)
│   │
│   ├── schemas/
│   │   └── patient.py             # All Pydantic models: PatientInput, PredictionResult,
│   │                              #   BiomarkerTrends, SimilarCase, HealingCategory
│   │
│   ├── ml/
│   │   ├── pipeline.py            # featurize() → 32-dim vector, compute_trends(),
│   │   │                          #   classify_category(), risk flags, recommendations
│   │   ├── trainer.py             # Trains all 5 models with StratifiedKFold CV, saves .pkl
│   │   ├── inference.py           # run_inference() → per-model probabilities + ensemble
│   │   └── saved_models/          # Persisted .pkl files (scaler + 5 models + cv_scores)
│   │
│   ├── rag/
│   │   ├── embedder.py            # Loads MiniLM, embed_patient(), embed_text()
│   │   ├── vector_store.py        # ChromaDB client, initialize(), add_case(),
│   │   │                          #   query_cases(), seeds real_patients.csv on first run
│   │   ├── retriever.py           # retrieve_similar_cases() → list[SimilarCase]
│   │   ├── tavily_search.py       # search_medical_literature() via Tavily API
│   │   └── llm_explainer.py       # GPT-4o clinical narrative, fallback if no API key
│   │
│   ├── mcp_server/
│   │   ├── server.py              # MCP JSON-RPC 2.0 router (/mcp/manifest, /mcp/call)
│   │   └── tools.py               # 4 MCP tool handlers + dispatch()
│   │
│   ├── routers/
│   │   ├── prediction.py          # POST /predict, /biomarker-trends, GET /models
│   │   ├── rag.py                 # POST /similar-cases, /ingest-case, GET /stats
│   │   └── mcp.py                 # Re-exports MCP server router
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── index.html                 # Landing page (7 sections describing the platform)
│   ├── dashboard.html             # Doctor input form + full results panel
│   ├── style.css                  # Design system: CSS variables, cards, gauge, charts
│   └── app.js                     # Form collection, fetch, Chart.js rendering, gauge SVG
│
├── data/
│   ├── real_patients.csv          # 30 real de-identified patients (used for ChromaDB)
│   ├── sample_patients.csv        # 500 synthetic patients (used to train ML models)
│   └── generate_synthetic.py      # Script to regenerate synthetic training data
│
├── chroma_db/                     # ChromaDB persistent store (auto-created on first run)
│
├── .env.example                   # Template for environment variables
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- pip

### 1. Clone / navigate to the project

```bash
cd fracture-healing-tool
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

If your system blocks pip (PEP 668), add `--break-system-packages` or use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...        # Required for GPT-4o clinical explanations
TAVILY_API_KEY=tvly-...      # Required for medical literature search
```

> **Without API keys** the system still works — ML prediction, biomarker trends, and similar case retrieval all function. Only the clinical explanation falls back to a rule-based summary.

### 4. Generate synthetic training data (first time only)

```bash
python3 data/generate_synthetic.py
```

This writes `data/sample_patients.csv` (500 rows) used to train the ML models.

### 5. Train the ML models (first time only)

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

> Steps 4 and 5 only need to be run once. On subsequent starts, the server loads the saved `.pkl` files directly.

### 6. Start the server

```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

On first startup, ChromaDB is automatically seeded with all 30 real patients from `data/real_patients.csv` and 10 medical literature chunks. This takes ~30 seconds.

### 7. Open in browser

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
        │
        │  POST /api/v1/prediction/predict  (JSON: PatientInput)
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Feature Engineering  (pipeline.py)             │
│                                                         │
│  PatientInput → 32-dimensional numpy feature vector     │
│                                                         │
│  Raw features (21):                                     │
│    age, gender, fracture_location                       │
│    bsap/alp/p1np × [day1, week3, week6]                 │
│    calcium/phosphorus × [day1, week3, week6]            │
│    callus × [day1, week3, week6]                        │
│                                                         │
│  Engineered features (11):                              │
│    bsap_delta, alp_delta, p1np_delta (day1→week6)       │
│    ca_delta, phos_delta                                 │
│    callus_d1_w3, callus_w3_w6, callus_d1_w6            │
│    bsap_alp_ratio_w6                                    │
│    ca_phos_product_w6                                   │
│    callus_growth_rate (mm²/day)                         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: ML Inference  (inference.py)                   │
│                                                         │
│  feature_vector → StandardScaler → 5 classifiers        │
│                                                         │
│  Each model returns P(Poor), P(Moderate), P(Good)       │
│  Best model (SVM by CV score) chosen as primary result  │
│  Weighted ensemble across all 5 models also computed    │
│                                                         │
│  Output:                                                │
│    healing_probability = P(Good) from best model        │
│    predicted_category  = argmax of best model proba     │
│    all_model_scores    = {model: P(Good), ...}          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Trend Analysis  (pipeline.py)                  │
│                                                         │
│  For each marker, compute:                              │
│    trend list: [day1, week3, week6]                     │
│    delta_pct:  % change from day1 to week6              │
│                                                         │
│  Rule-based trend narrative:                            │
│    "BSAP rising (+34%), suggesting active bone          │
│     formation; callus growing well (+847%)"             │
│                                                         │
│  Risk flags (rule-based, not ML):                       │
│    BSAP declining → reduced osteoblast activity         │
│    Callus growth < 20% → risk of delayed union          │
│    P1NP trend negative → impaired collagen synthesis    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: RAG — Similar Case Retrieval  (retriever.py)   │
│                                                         │
│  1. Convert patient to natural-language summary text    │
│     "41yo male with tibia fracture. BSAP day1=23..."   │
│                                                         │
│  2. Embed with MiniLM-L6-v2 → 384-dim vector           │
│                                                         │
│  3. Cosine similarity search in ChromaDB                │
│     (30 real patients stored as embeddings)             │
│                                                         │
│  4. Return top-3 most similar cases with:              │
│     patient_name, phone_no, age, gender,                │
│     fracture_location, callus_w6, outcome,              │
│     similarity_score                                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: Tavily Web Search  (tavily_search.py)          │
│                                                         │
│  Query: "{fracture_location} fracture healing           │
│          prognosis BSAP ALP P1NP callus {category}"    │
│                                                         │
│  Returns 3–5 snippets from medical journals/databases   │
│  (skipped gracefully if TAVILY_API_KEY not set)         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6: GPT-4o Clinical Explanation  (llm_explainer)   │
│                                                         │
│  Prompt contains:                                       │
│    • Patient demographics + biomarker values            │
│    • ML probabilities from all 5 models                 │
│    • Biomarker trend summary (% changes)                │
│    • 3 similar historical cases                         │
│    • Tavily literature snippets                         │
│                                                         │
│  Output: 3–4 paragraph clinical narrative               │
│  (framed as "prediction", not diagnosis)                │
│                                                         │
│  Fallback: rule-based summary if no OPENAI_API_KEY      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 7: Response Assembly  (prediction.py router)      │
│                                                         │
│  PredictionResult {                                     │
│    healing_probability,  healing_probability_pct,       │
│    healing_category,     model_used,                    │
│    confidence_scores,    biomarker_trends,              │
│    similar_cases,        clinical_explanation,          │
│    risk_flags,           recommendations                │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Dashboard renders:
  • Animated SVG gauge (healing %)
  • Category badge (Good / Moderate / Poor)
  • All-model score bar chart
  • 4 Chart.js line charts (BSAP, ALP, P1NP, Callus)
  • Similar patients table (name, phone, similarity %)
  • GPT-4o clinical narrative
  • Risk flags + recommendations
```

### Startup Flow (server boot)

```
uvicorn backend.main:app
        │
        ├── Load ML Pipeline
        │     • Check backend/ml/saved_models/ for .pkl files
        │     • If found → load scaler + 5 models (fast, ~0.5s)
        │     • If not found → train on data/sample_patients.csv (~60s)
        │
        ├── Initialise Vector Store
        │     • Connect to ChromaDB at ./chroma_db/
        │     • If patient_cases collection is empty:
        │         Read data/real_patients.csv (30 patients)
        │         Embed each patient summary with MiniLM
        │         Store in ChromaDB
        │     • If literature collection is empty:
        │         Embed 10 medical knowledge chunks
        │         Store in ChromaDB
        │
        └── Server ready → http://0.0.0.0:8000
```

### ChromaDB Seeding Flow (first run only)

```
data/real_patients.csv (30 rows)
        │
        ▼
For each patient row:
  Build summary text:
    "Patient: Satya Prakash Singh (Phone: 9718184829).
     41yo male with tibia fracture.
     BSAP day1=23.1→week6=34.7. ALP day1=59.6→week6=142.3.
     Callus week6=125. Outcome: Moderate."
        │
        ▼
  MiniLM-L6-v2 encodes summary → 384-dim float vector
        │
        ▼
  ChromaDB stores: { id, embedding, metadata, document }
        │
        ▼
30 cases stored. Cosine similarity search ready.
```

---

## API Reference

### Prediction

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/prediction/predict` | Full pipeline — returns `PredictionResult` |
| POST | `/api/v1/prediction/biomarker-trends` | Trend analysis only (no LLM, fast) |
| GET | `/api/v1/prediction/models` | Lists models and CV scores |

**Sample request body for `/predict`:**

```json
{
  "age": 41,
  "gender": "male",
  "fracture_location": "tibia",
  "biomarkers_day1":  { "bsap": 23.1, "alp": 59.6,  "p1np": 51.8  },
  "biomarkers_week3": { "bsap": 26.2, "alp": 121.5, "p1np": 70.7  },
  "biomarkers_week6": { "bsap": 34.7, "alp": 142.3, "p1np": 102.1 },
  "minerals_day1":    { "calcium": 9.85, "phosphorus": 4.56 },
  "minerals_week3":   { "calcium": 9.17, "phosphorus": 2.91 },
  "minerals_week6":   { "calcium": 9.60, "phosphorus": 3.83 },
  "callus_d1": 6.4,
  "callus_w3": 42.4,
  "callus_w6": 125.1
}
```

Valid `fracture_location` values: `femur`, `tibia`, `radius`, `ulna`, `humerus`, `fibula`, `pelvis`, `vertebra`

### RAG

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/rag/similar-cases` | Returns top-3 similar patients from ChromaDB |
| POST | `/api/v1/rag/ingest-case` | Adds a new case to the vector store |
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

Register this server in Claude Desktop's `claude_desktop_config.json`:

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

> *"Predict healing for a 45-year-old male with tibia fracture and BSAP day1=28, week3=36, week6=44, ALP day1=75 …"*

Claude will invoke `predict_fracture_healing`, chain it with `explain_prediction`, and return a structured clinical report without any additional code.

---

## Dataset

### Real Patients (`data/real_patients.csv`)

- **30 patients** from actual clinical records
- Fracture types: Tibia (12), Femur (8), Humerus (5), Ulna (3), Radius (2)
- Gender: Male (21), Female (9)
- All patients healed to Moderate or Good category (no Poor cases in this cohort)
- Used exclusively for ChromaDB similar-case retrieval (RAG)
- Contains: `patient_name`, `phone_no`, `age`, `gender`, `fracture_location`, full biomarker panel, `healing_category`

### Synthetic Data (`data/sample_patients.csv`)

- **500 patients** generated with realistic statistical distributions
- Distribution: Good (200), Moderate (175), Poor (125)
- Used only for ML model training — not shown to users
- Includes Poor category cases (absent in real data) to give models exposure to all three classes
- Regenerate anytime: `python3 data/generate_synthetic.py`

### Why Two Datasets?

The real dataset has only 30 patients — too few for reliable 5-fold cross-validation (only 6 samples per fold). Synthetic data augments this for training while the real data provides authentic case context for RAG retrieval.
