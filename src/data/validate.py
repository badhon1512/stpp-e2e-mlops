import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LABELS, TARGET_COLUMN, TEST_FILE, TEXT_COLUMN, TRAIN_FILE, VALIDATION_FILE, VALIDATION_REPORT_FILE


def _check_split(name: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    return {
        "split": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "empty_text_rows": int(df[TEXT_COLUMN].fillna("").astype(str).str.strip().eq("").sum()),
        "unknown_labels": sorted(set(df[TARGET_COLUMN].dropna().astype(str).str.lower()) - set(LABELS)),
        "label_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }


def main() -> dict:
    report = {
        "train": _check_split("train", TRAIN_FILE),
        "validation": _check_split("validation", VALIDATION_FILE),
        "test": _check_split("test", TEST_FILE),
    }
    VALIDATION_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

