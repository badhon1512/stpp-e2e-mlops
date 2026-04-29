# Roadmap

## Phase 1

- Scaffold project structure.
- Define dataset target, labels, and configs.
- Set up backend and frontend starter apps.

## Phase 2

- Implement Kaggle ingestion.
- Validate and preprocess English-only text data.
- Save processed train, validation, and test splits.

## Phase 3

- Train a TF-IDF + Logistic Regression baseline with candidate search.
- Save the sklearn model artifact, validation/test metrics, and selected candidate.
- Track runs and register the champion model in MLflow when tracking is configured.
- Compare with a transformer model later only if the baseline does not meet quality or latency goals.

## Phase 4

- Load the trained local sklearn model or MLflow champion model in the backend.
- Add prediction logging, tests, and request validation.
- Connect the Next.js frontend to real backend responses through an API proxy.

## Phase 5

- Dockerize backend and frontend.
- Deploy on AWS.
- Add CI checks for pytest and DVC pipeline reproduction.
- Add monitoring, drift reports, and retraining jobs.
- Promote models through MLflow aliases instead of hardcoded versions.
