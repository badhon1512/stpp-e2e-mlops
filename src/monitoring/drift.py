from config import REPORTS_DIR


def monitoring_plan() -> dict:
    return {
        "report_dir": str(REPORTS_DIR),
        "signals": [
            "request_latency",
            "prediction_distribution",
            "confidence_distribution",
            "text_length_drift",
            "missing_input_rate",
        ],
        "todo": "generate drift reports on a schedule after deployment",
    }


if __name__ == "__main__":
    print(monitoring_plan())

