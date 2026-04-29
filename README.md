# Support Ticket Priority Classification Platform

An end-to-end NLP MLOps project for predicting support ticket priority from ticket text. This project uses a compact English-only preprocessing pipeline, lightweight sklearn-based model selection, a FastAPI backend, MLflow tracking/registry, and a TypeScript frontend.

## Problem

Input:
- `subject`
- `body`

Output:
- `low`
- `medium`
- `high`

The starter dataset is:
- `tobiasbueck/multilingual-customer-support-tickets`

Version 1 will focus on English-only records and a single target column: `priority`.

## Current Stack

- Backend: FastAPI
- Frontend: Next.js + TypeScript
- Model: TF-IDF + Logistic Regression candidate search
- Data: pandas + scikit-learn split pipeline
- Tracking and registry: MLflow
- Pipeline orchestration: DVC
- Testing: pytest
- Deployment target: AWS

## Run

Inside this project directory:

```powershell
uv sync
uv run ticket-ingest
uv run ticket-validate
uv run ticket-train
uv run python main.py
```

Run the frontend in another terminal:

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

The Next.js app runs at `http://localhost:3000` and proxies prediction requests to the FastAPI backend configured by `frontend/.env.local`.

Or run the scripts directly:

```powershell
uv run python src\data\ingest.py
uv run python src\data\validate.py
uv run python src\training\train.py
uv run python main.py
```

If the console scripts such as `ticket-api` are not installed in your virtual environment yet, use the `uv run python ...` commands above.

Run the full reproducible data/training pipeline with DVC:

```powershell
dvc repro
```

## Docker

Build and run the full app locally:

```powershell
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Backend health: `http://localhost:8000/health`
- Model info: `http://localhost:8000/model-info`

The frontend service uses `BACKEND_API_URL=http://backend:8000` inside Docker Compose. `backend` is the Docker Compose service name, not a folder path.

Stop the app:

```powershell
docker compose down
```

Rebuild without cache:

```powershell
docker compose build --no-cache
```

## Outputs

- Processed data: `data/processed/train.csv`, `validation.csv`, `test.csv`
- Reports: `artifacts/reports/*.json`
- Model: `artifacts/models/ticket_priority_sklearn.joblib`
- DVC pipeline: `dvc.yaml`
- Parameters: `params.yaml`

## Layout

- `backend/`: API service
- `frontend/`: TypeScript client
- `tests/`: basic data and API tests
- `src/data/`: ingestion and validation
- `src/training/`: training and evaluation
- `src/inference/`: local model loading and prediction helpers
- `src/monitoring/`: monitoring jobs and reports
- `configs/`: project settings
- `docs/`: roadmap and deployment notes
