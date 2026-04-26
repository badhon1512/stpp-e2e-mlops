import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    APP_HOST,
    APP_PORT,
    APP_RELOAD,
    LOG_DIR,
    LOG_FILE,
    LOCAL_MODEL_FILE,
    MODEL_ALIAS,
    PROJECT_DISPLAY_NAME,
    PROJECT_NAME,
    REGISTERED_MODEL_NAME,
    TEMP_DIR,
)
from src.inference.sklearn_predictor import LocalSklearnPredictor

try:
    import mlflow
    import mlflow.pyfunc
except ImportError:  # pragma: no cover
    mlflow = None


load_dotenv()

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)

logger = logging.getLogger(PROJECT_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)


class PredictionRequest(BaseModel):
    subject: str = Field(..., min_length=3)
    body: str = Field(..., min_length=5)


class PredictionResponse(BaseModel):
    predicted_priority: str
    confidence: float
    model_uri: str
    latency_ms: float


class StubPriorityModel:
    def predict(self, payload: dict[str, str]) -> dict[str, Any]:
        text = f"{payload['subject']} {payload['body']}".lower()

        high_keywords = ("outage", "urgent", "critical", "breach", "payment failed", "down")
        medium_keywords = ("error", "issue", "cannot", "slow", "failed", "bug")

        if any(keyword in text for keyword in high_keywords):
            return {"label": "high", "score": 0.91}
        if any(keyword in text for keyword in medium_keywords):
            return {"label": "medium", "score": 0.76}
        return {"label": "low", "score": 0.63}


class MLflowPriorityModel:
    def __init__(self, model):
        self.model = model

    def predict(self, payload: dict[str, str]) -> dict[str, Any]:
        frame = pd.DataFrame([payload])
        print("model_input_frame", frame)
        result = self.model.predict(frame)

        if hasattr(result, "to_dict"):
            records = result.to_dict(orient="records")
            return records[0]
        if isinstance(result, list):
            return result[0]
        if isinstance(result, dict):
            return result
        raise TypeError("Unsupported MLflow prediction output format.")


def load_priority_model():
    if mlflow is not None and os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/5"

        try:
            print(f"Attempting to load MLflow model from {model_uri}...")
            model = MLflowPriorityModel(mlflow.pyfunc.load_model(model_uri))
            logger.info("model_loaded | model_uri=%s", model_uri)
            print("MLflow model loaded successfully.")
            return model, model_uri
        except Exception:
            logger.exception("mlflow_model_load_failed | using_next_fallback=true")

    if LOCAL_MODEL_FILE.exists():
        try:
            model = LocalSklearnPredictor(LOCAL_MODEL_FILE)
            model_uri = str(LOCAL_MODEL_FILE)
            logger.info("model_loaded | model_uri=%s", model_uri)
            return model, model_uri
        except Exception:
            logger.exception("local_model_load_failed | using_stub_model=true")

    return StubPriorityModel(), "stub://ticket-priority-heuristic"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_uri = load_priority_model()
    yield


app = FastAPI(title=PROJECT_DISPLAY_NAME, lifespan=lifespan)


@app.get("/")
def home():
    return {
        "project": PROJECT_NAME,
        "display_name": PROJECT_DISPLAY_NAME,
        "task": "ticket-priority-classification",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "active_model_uri": app.state.model_uri,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        payload = {"subject": request.subject, "body": request.body}
        result = app.state.model.predict(payload)

        if isinstance(result, list):
            result = result[0]

        latency_ms = round((time.time() - start_time) * 1000, 2)
        predicted_priority = result.get("label", "unknown")
        confidence = float(result.get("score", 0.0))

        logger.info(
            "prediction_success | model_uri=%s | latency_ms=%s | subject_length=%s | body_length=%s | prediction=%s | confidence=%s",
            app.state.model_uri,
            latency_ms,
            len(request.subject),
            len(request.body),
            predicted_priority,
            confidence,
        )

        return PredictionResponse(
            predicted_priority=predicted_priority,
            confidence=confidence,
            model_uri=app.state.model_uri,
            latency_ms=latency_ms,
        )
    except Exception as error:
        logger.exception("prediction_failed | model_uri=%s", app.state.model_uri)
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=APP_HOST, port=APP_PORT, reload=APP_RELOAD)
