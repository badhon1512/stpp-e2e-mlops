from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def load_params() -> dict[str, Any]:
    if not PARAMS_FILE.exists():
        return {}

    with PARAMS_FILE.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("params.yaml must contain a top-level mapping/object.")
    return loaded


def get_param(params: dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = params
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

