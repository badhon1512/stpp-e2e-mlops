import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    C_VALUE,
    EXPERIMENT_NAME,
    LOCAL_MODEL_FILE,
    MAX_FEATURES,
    MAX_ITER,
    MODEL_ALIAS,
    NGRAM_RANGE,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
    TEST_FILE,
    TEXT_COLUMN,
    TRAIN_FILE,
    TRAINING_REPORT_FILE,
    VALIDATION_FILE,
    VALIDATION_REPORT_FILE,
)
from src.data.ingest import main as prepare_data
from src.data.validate import main as validate_data
from src.inference.sklearn_predictor import LocalSklearnPredictor
from src.utils.params import get_param, load_params

try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover
    mlflow = None
    MlflowClient = None


load_dotenv()

DEFAULT_CANDIDATE_CONFIGS = [
    {
        "name": "word_unigram_bigram",
        "max_features": MAX_FEATURES,
        "ngram_range": NGRAM_RANGE,
        "c_value": C_VALUE,
        "stop_words": "english",
        "min_df": 2,
        "sublinear_tf": False,
    },
    {
        "name": "word_unigram_bigram_high_c",
        "max_features": 40000,
        "ngram_range": (1, 2),
        "c_value": 8.0,
        "stop_words": "english",
        "min_df": 2,
        "sublinear_tf": False,
    },
    {
        "name": "word_unigram_bigram_very_high_c",
        "max_features": 50000,
        "ngram_range": (1, 2),
        "c_value": 12.0,
        "stop_words": "english",
        "min_df": 2,
        "sublinear_tf": False,
    },
    {
        "name": "word_bigram_sublinear",
        "max_features": 50000,
        "ngram_range": (1, 2),
        "c_value": 6.0,
        "stop_words": "english",
        "min_df": 3,
        "sublinear_tf": True,
    },
    {
        "name": "word_trigram",
        "max_features": 50000,
        "ngram_range": (1, 3),
        "c_value": 6.0,
        "stop_words": "english",
        "min_df": 2,
        "sublinear_tf": False,
    },
    {
        "name": "word_trigram_sublinear",
        "max_features": 60000,
        "ngram_range": (1, 3),
        "c_value": 8.0,
        "stop_words": "english",
        "min_df": 3,
        "sublinear_tf": True,
    },
    {
        "name": "word_no_stopwords",
        "max_features": 35000,
        "ngram_range": (1, 2),
        "c_value": 4.0,
        "stop_words": None,
        "min_df": 2,
        "sublinear_tf": False,
    },
    {
        "name": "word_low_min_df",
        "max_features": 60000,
        "ngram_range": (1, 2),
        "c_value": 4.0,
        "stop_words": "english",
        "min_df": 1,
        "sublinear_tf": True,
    },
    {
        "name": "char_wb_3_5",
        "analyzer": "char_wb",
        "max_features": 70000,
        "ngram_range": (3, 5),
        "c_value": 6.0,
        "stop_words": None,
        "min_df": 3,
        "sublinear_tf": True,
    },
    {
        "name": "char_wb_4_6",
        "analyzer": "char_wb",
        "max_features": 90000,
        "ngram_range": (4, 6),
        "c_value": 8.0,
        "stop_words": None,
        "min_df": 3,
        "sublinear_tf": True,
    },
]


class TicketPriorityPyfuncModel(mlflow.pyfunc.PythonModel if mlflow is not None else object):
    def load_context(self, context):
        self.predictor = LocalSklearnPredictor(Path(context.artifacts["model_file"]))

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict_dataframe(model_input)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists() or not TEST_FILE.exists():
        prepare_data()
    validate_data()
    return pd.read_csv(TRAIN_FILE), pd.read_csv(VALIDATION_FILE), pd.read_csv(TEST_FILE)


def build_model(config: dict) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer=config.get("analyzer", "word"),
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words=config["stop_words"],
                    ngram_range=config["ngram_range"],
                    max_features=config["max_features"],
                    min_df=config.get("min_df", 1),
                    sublinear_tf=config.get("sublinear_tf", False),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    C=config["c_value"],
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def load_candidate_configs() -> list[dict]:
    params = load_params()
    configured_candidates = get_param(params, "train.candidate_configs", DEFAULT_CANDIDATE_CONFIGS)
    if not isinstance(configured_candidates, list) or not configured_candidates:
        raise ValueError("train.candidate_configs in params.yaml must be a non-empty list.")
    normalized_candidates = []
    for candidate in configured_candidates:
        normalized = dict(candidate)
        normalized["ngram_range"] = tuple(normalized["ngram_range"])
        normalized_candidates.append(normalized)
    return normalized_candidates


def evaluate_model(model: Pipeline, df: pd.DataFrame, label_names: list[str], split_name: str) -> dict:
    predictions = model.predict(df[TEXT_COLUMN])
    probabilities = model.predict_proba(df[TEXT_COLUMN])
    return {
        "split": split_name,
        "accuracy": round(float(accuracy_score(df[TARGET_COLUMN], predictions)), 4),
        "macro_f1": round(float(f1_score(df[TARGET_COLUMN], predictions, average="macro")), 4),
        "classification_report": classification_report(
            df[TARGET_COLUMN], predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            df[TARGET_COLUMN], predictions, labels=label_names
        ).tolist(),
        "average_confidence": round(float(probabilities.max(axis=1).mean()), 4),
    }


def _configure_mlflow() -> bool:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow is None or not tracking_uri:
        return False

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return True


def _set_model_alias(run_id: str) -> None:
    if MlflowClient is None:
        return

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    matching_versions = [version for version in versions if version.run_id == run_id]
    if not matching_versions:
        return

    latest_version = max(matching_versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, latest_version.version)


def _log_to_mlflow(summary: dict) -> None:
    if not _configure_mlflow():
        return

    with mlflow.start_run(run_name="tfidf-logreg-ticket-priority-training") as run:
        mlflow.log_params(
            {
                "model_type": "tfidf_logistic_regression",
                "candidate_name": summary["best_candidate"]["name"],
                "max_features": summary["best_candidate"]["max_features"],
                "ngram_range": str(tuple(summary["best_candidate"]["ngram_range"])),
                "max_iter": MAX_ITER,
                "c_value": summary["best_candidate"]["c_value"],
                "stop_words": str(summary["best_candidate"]["stop_words"]),
                "analyzer": summary["best_candidate"].get("analyzer", "word"),
                "min_df": summary["best_candidate"].get("min_df", 1),
                "sublinear_tf": summary["best_candidate"].get("sublinear_tf", False),
                "text_column": TEXT_COLUMN,
            }
        )

        mlflow.log_metric("final_validation_macro_f1", float(summary["validation"]["macro_f1"]))
        mlflow.log_metric("final_validation_accuracy", float(summary["validation"]["accuracy"]))
        mlflow.log_metric("final_test_macro_f1", float(summary["test"]["macro_f1"]))
        mlflow.log_metric("final_test_accuracy", float(summary["test"]["accuracy"]))
        mlflow.log_metric("validation_average_confidence", float(summary["validation"]["average_confidence"]))
        mlflow.log_metric("test_average_confidence", float(summary["test"]["average_confidence"]))

        for artifact in [TRAINING_REPORT_FILE, VALIDATION_REPORT_FILE]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TicketPriorityPyfuncModel(),
            artifacts={"model_file": str(LOCAL_MODEL_FILE)},
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        summary["mlflow"] = {
            "tracking_uri": mlflow.get_tracking_uri(),
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "model_alias": MODEL_ALIAS,
        }
        _set_model_alias(run.info.run_id)


def main() -> dict:
    training_start = time.time()
    progress = tqdm(total=7, desc="Training pipeline")
    candidate_configs = load_candidate_configs()

    progress.set_description("Loading data")
    train_df, validation_df, test_df = load_data()
    label_names = sorted(train_df[TARGET_COLUMN].unique().tolist())
    progress.update(1)

    progress.set_description("Preparing candidates")
    print("Starting TF-IDF + Logistic Regression model selection")
    print(
        f"Training rows={len(train_df)} | "
        f"Validation rows={len(validation_df)} | "
        f"Test rows={len(test_df)}"
    )
    progress.update(1)

    progress.set_description("Selecting best candidate")
    candidate_summaries = []
    best_candidate = None
    best_model = None
    best_validation = None

    for candidate in tqdm(candidate_configs, desc="Candidate search", leave=False):
        print(
            f"Candidate={candidate['name']} | "
            f"analyzer={candidate.get('analyzer', 'word')} | "
            f"max_features={candidate['max_features']} | "
            f"ngram_range={candidate['ngram_range']} | "
            f"C={candidate['c_value']} | "
            f"stop_words={candidate['stop_words']} | "
            f"min_df={candidate.get('min_df', 1)} | "
            f"sublinear_tf={candidate.get('sublinear_tf', False)}"
        )
        model = build_model(candidate)
        model.fit(train_df[TEXT_COLUMN], train_df[TARGET_COLUMN])
        validation_metrics = evaluate_model(model, validation_df, label_names, "validation")
        candidate_result = {
            **candidate,
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        candidate_summaries.append(candidate_result)

        if best_validation is None or validation_metrics["macro_f1"] > best_validation["macro_f1"]:
            best_candidate = candidate
            best_model = model
            best_validation = validation_metrics

    progress.update(1)
    print(
        f"Best candidate={best_candidate['name']} | "
        f"val_f1={best_validation['macro_f1']:.4f} | "
        f"val_acc={best_validation['accuracy']:.4f}"
    )

    progress.set_description("Refitting best candidate")
    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_model = build_model(best_candidate)
    final_model.fit(combined_train_df[TEXT_COLUMN], combined_train_df[TARGET_COLUMN])
    progress.update(1)

    progress.set_description("Evaluating validation")
    validation_metrics = evaluate_model(best_model, validation_df, label_names, "validation")
    progress.update(1)

    progress.set_description("Evaluating test")
    test_metrics = evaluate_model(final_model, test_df, label_names, "test")
    progress.update(1)

    progress.set_description("Saving artifacts")
    LOCAL_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, LOCAL_MODEL_FILE)

    summary = {
        "model": "tfidf_logistic_regression",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "max_iter": MAX_ITER,
        "candidate_results": candidate_summaries,
        "best_candidate": {
            **best_candidate,
            "ngram_range": list(best_candidate["ngram_range"]),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "model_file": str(LOCAL_MODEL_FILE),
        "training_time_seconds": round(time.time() - training_start, 2),
    }

    TRAINING_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_REPORT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log_to_mlflow(summary)
    TRAINING_REPORT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress.update(1)
    progress.close()

    print(
        f"Training complete in {summary['training_time_seconds']}s | "
        f"final_val_f1={validation_metrics['macro_f1']:.4f} | "
        f"final_test_f1={test_metrics['macro_f1']:.4f}"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
