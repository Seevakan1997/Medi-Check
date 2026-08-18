# MedCheck — Healthcare Misinformation Detector

A multi-layer AI system that detects healthcare misinformation using:

- **BioBERT** deep learning classifier (Layer 1)
- **RAG** knowledge base from WHO/NHS/CDC (Layer 2)
- **LLaMA 3** local LLM reasoning (Layer 3)
- **Full provenance tracking** and web dashboard

---

## Project Structure

```
project/
├── data/
│   └── health_fact.csv        ← your dataset goes here
├── step1_train.py             ← train the DL model
├── step2_build_rag.py         ← build the RAG knowledge base
├── step3_run.py               ← run the pipeline (terminal)
├── app.py                     ← Flask API server
├── dashboard.html             ← web dashboard
├── requirements.txt
└── README.md
```

---

## Setup (do this once)

### 1. Create virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Install Ollama and pull LLaMA 3

Download Ollama from: https://ollama.com/download

Then in a terminal:

```bash
ollama pull llama3
```

---

## Running the System

### Step 1 — Prepare your dataset

Make sure `data/health_fact.csv` exists with columns: `text`, `label`

If you haven't downloaded it yet:

```bash
python data/download_data.py
```

### Step 2 — Train the DL model (~30 min on CPU)

```bash
python step1_train.py
```

Saves: `model.pt`

### Step 3 — Build the RAG index (~1 min)

```bash
python step2_build_rag.py
```

Creates: `chroma_db/`

### Step 4 — Start Ollama (in a separate terminal, keep it running)

```bash
ollama serve
```

### Step 5a — Run in terminal (interactive)

```bash
python step3_run.py
```

### Step 5b — Run single claim

```bash
python step3_run.py --text "Bleach cures COVID-19."
```

### Step 5c — Run with dashboard (recommended)

```bash
python app.py
```

Then open: http://localhost:5000

---

## How it works

```
Input claim
    ↓
[Layer 1] BioBERT classifier → risk score (0-100%)
    ↓
[Layer 2] ChromaDB RAG search → WHO/NHS/CDC evidence
    ↓
[Layer 3] LLaMA 3 (local) → verdict + reasoning + recommendation
    ↓
Provenance record → audit trail saved
    ↓
Dashboard → visual result with gauge, RAG bars, history chart
```

---

## API Endpoints (when running app.py)

| Method | Endpoint     | Description                     |
| ------ | ------------ | ------------------------------- |
| GET    | /            | Serves the dashboard            |
| GET    | /api/health  | Checks all components are ready |
| POST   | /api/analyse | Runs the full pipeline          |

### Example API call

```bash
curl -X POST http://localhost:5000/api/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "Vaccines cause autism."}'
```

---

## Expanding the knowledge base

Add more entries to the `KNOWLEDGE_BASE` list in `step2_build_rag.py`:

```python
{
    "id":       "unique-id",
    "claim":    "The false claim text.",
    "verdict":  "misinformation",
    "source":   "WHO / NHS / CDC",
    "url":      "https://...",
    "rationale":"Why this is misinformation."
}
```

Then rebuild the index:

```bash
python step2_build_rag.py
```

---

## Troubleshooting

| Error                           | Fix                                       |
| ------------------------------- | ----------------------------------------- |
| `model.pt not found`            | Run `python step1_train.py` first         |
| `chroma_db not found`           | Run `python step2_build_rag.py` first     |
| `Cannot connect to Ollama`      | Run `ollama serve` in a separate terminal |
| `health_fact.csv not found`     | Run `python data/download_data.py` first  |
| `huggingface-hub version error` | Run `pip install transformers -U`         |
