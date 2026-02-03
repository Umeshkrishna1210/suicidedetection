# Suicide Risk (Signal) — Real-Time Research Prototype

This system outputs **risk signals** and **emotion signals** from social-media-like text. It is **not** a medical device and must not be used to diagnose or treat.

## Quickstart (Toy Data)

### 1) Setup

```powershell
cd "c:\umesh laptop\COLLEGE\ALL CODING\PROJECTS\MAAJOR PROJECT EXECUTION\suicide-risk-rt"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2) Train + Evaluate (Toy)

```powershell
python scripts\train.py --config configs\default.yaml --overrides data.source=toy
python scripts\evaluate.py --config configs\default.yaml --overrides data.source=toy
```

Artifacts:
- `models/` checkpoints
- `outputs/` metrics + plots

### 3) Build RAG Index (Toy) + Serve API

```powershell
python scripts\build_rag_index.py --config configs\default.yaml --overrides data.source=toy
python scripts\serve_api.py --config configs\default.yaml --overrides data.source=toy
```

API:
- `GET /health`
- `POST /predict`

### 4) Dashboard

```powershell
streamlit run scripts\dashboard.py
```

## Compare Experiments (No Retraining)

After you have runs under `outputs/` (e.g. from `scripts/run_experiments.py` or individual train/eval runs), generate a report-ready comparison table + plots:

```powershell
python scripts\aggregate_results.py
```

Outputs:
- `outputs/comparisons/comparison_table.csv`
- `outputs/comparisons/comparison_table.md`
- `outputs/comparisons/risk_f1_macro.png`
- `outputs/comparisons/emotion_f1_macro.png`

## External Dataset

This repo supports HuggingFace datasets and CSV/JSONL. **Before enabling any external dataset**, you must explicitly confirm:
- Dataset name
- Source (Kaggle / GitHub / official site)
- Expected size
- License or access requirements

See `configs/default.yaml` → `data.*`.

## Safety

See `SAFETY.md` for guardrails and required usage constraints.
