import pandas as pd

from src.data.ingest import prepare_dataframe


def test_prepare_dataframe_filters_to_english_and_builds_text():
    df = pd.DataFrame(
        [
            {"subject": "Billing outage", "body": "Payment page is down", "priority": "high", "language": "en"},
            {"subject": "Hallo", "body": "Bitte helfen", "priority": "low", "language": "de"},
            {"subject": "", "body": "Need access reset", "priority": "medium", "language": "en"},
        ]
    )

    processed, report = prepare_dataframe(df)

    assert len(processed) == 2
    assert processed["priority"].tolist() == ["high", "medium"]
    assert processed["text"].iloc[1] == "Need access reset"
    assert report["filtered_rows"] == 2
