from pathlib import Path

import joblib
import pandas as pd


INPUT_COLUMNS = ("subject", "body")
OUTPUT_COLUMNS = ("label", "score")


def _normalize_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def build_inference_frame(payloads: list[dict[str, str]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(payloads, pd.DataFrame):
        frame = payloads.copy()
    else:
        frame = pd.DataFrame(payloads)

    for column in INPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].map(_normalize_text)

    frame["text"] = (frame["subject"] + " " + frame["body"]).str.strip()
    return frame.loc[:, ["subject", "body", "text"]]


class LocalSklearnPredictor:
    def __init__(self, model_file: Path):
        self.model = joblib.load(model_file)

    def predict_dataframe(self, model_input: list[dict[str, str]] | pd.DataFrame) -> pd.DataFrame:
        frame = build_inference_frame(model_input)
        probabilities = self.model.predict_proba(frame["text"])
        classes = list(self.model.classes_)
        best_indices = probabilities.argmax(axis=1)

        return pd.DataFrame(
            {
                "label": [classes[index] for index in best_indices],
                "score": [
                    round(float(probabilities[row_index, class_index]), 4)
                    for row_index, class_index in enumerate(best_indices)
                ],
            },
            columns=list(OUTPUT_COLUMNS),
        )

    def predict(self, payload: dict[str, str]) -> dict[str, float | str]:
        return self.predict_dataframe([payload]).iloc[0].to_dict()
