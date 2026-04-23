import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_LANGUAGE,
    EDA_REPORT_FILE,
    LANGUAGE_COLUMN,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TEST_SIZE,
    TEXT_COLUMN,
    TRAIN_FILE,
    VALIDATION_FILE,
    VALIDATION_SIZE,
)
from src.utils.params import get_param, load_params


def _clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def load_raw_data(raw_data_file: Path) -> pd.DataFrame:
    if not raw_data_file.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_file}")
    return pd.read_csv(raw_data_file)


def prepare_dataframe(df: pd.DataFrame, language: str = DEFAULT_LANGUAGE) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["subject"] = df["subject"].apply(_clean_text)
    df["body"] = df["body"].apply(_clean_text)
    df[LANGUAGE_COLUMN] = df[LANGUAGE_COLUMN].astype(str).str.lower().str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.lower().str.strip()

    filtered_df = df[df[LANGUAGE_COLUMN] == language].copy()
    filtered_df[TEXT_COLUMN] = (filtered_df["subject"] + " " + filtered_df["body"]).str.replace(r"\s+", " ", regex=True).str.strip()
    filtered_df = filtered_df[filtered_df[TEXT_COLUMN] != ""].copy()
    filtered_df = filtered_df[filtered_df[TARGET_COLUMN].isin(["low", "medium", "high"])].copy()

    processed = filtered_df[["subject", "body", TEXT_COLUMN, TARGET_COLUMN]].reset_index(drop=True)

    report = {
        "source_rows": int(len(df)),
        "language": language,
        "filtered_rows": int(len(processed)),
        "priority_distribution": processed[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "missing_subject_rows": int((filtered_df["subject"] == "").sum()),
        "missing_body_rows": int((filtered_df["body"] == "").sum()),
        "average_text_length": round(float(processed[TEXT_COLUMN].str.len().mean()), 2),
        "median_text_length": round(float(processed[TEXT_COLUMN].str.len().median()), 2),
    }
    return processed, report


def split_and_save(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    validation_ratio = validation_size / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_df,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=train_df[TARGET_COLUMN],
    )

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    EDA_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    return {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
    }


def main() -> dict:
    params = load_params()
    raw_data_file = PROJECT_ROOT / get_param(
        params, "data.raw_data_file", str(RAW_DATA_FILE.relative_to(PROJECT_ROOT))
    )
    language = get_param(params, "data.language", DEFAULT_LANGUAGE)
    test_size = float(get_param(params, "split.test_size", TEST_SIZE))
    validation_size = float(get_param(params, "split.validation_size", VALIDATION_SIZE))
    random_state = int(get_param(params, "split.random_state", RANDOM_STATE))

    raw_df = load_raw_data(raw_data_file)
    processed_df, report = prepare_dataframe(raw_df, language=language)
    split_sizes = split_and_save(
        processed_df,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    report.update(split_sizes)
    report["raw_data_file"] = str(raw_data_file)
    report["test_size"] = test_size
    report["validation_size"] = validation_size
    report["random_state"] = random_state
    EDA_REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
