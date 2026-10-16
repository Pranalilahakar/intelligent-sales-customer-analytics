
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "customer_segments.csv"
OUTPUT_FILE = PROCESSED_DIR / "customer_priority_scores.csv"


def percentile_score(series, higher_is_better=True):
    """Return percentile scores from 0 to 100."""
    percentile = series.rank(pct=True, method="average") * 100

    if not higher_is_better:
        percentile = 100 - percentile

    return percentile


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Segment_Name",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["CustomerID"].isna().any():
        raise ValueError("Missing CustomerID values found.")

    if df["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customer records found.")

    for column in ["Recency", "Frequency", "Monetary"]:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    if (
        (df["Recency"] < 0).any()
        or (df["Frequency"] < 0).any()
        or (df["Monetary"] < 0).any()
    ):
        raise ValueError(
            "RFM inputs contain negative values."
        )

    # Higher recency means a longer time since purchase,
    # so lower Recency receives a higher score.
    df["Recency_Score"] = percentile_score(
        df["Recency"], higher_is_better=False
    )

    df["Frequency_Score"] = percentile_score(
        df["Frequency"], higher_is_better=True
    )

    df["Monetary_Score"] = percentile_score(
        df["Monetary"], higher_is_better=True
    )

    # Transparent weights: recency 40%, frequency 30%,
    # monetary value 30%.
    df["Priority_Score"] = (
        0.40 * df["Recency_Score"]
        + 0.30 * df["Frequency_Score"]
        + 0.30 * df["Monetary_Score"]
    ).round(2)

    df["Priority_Category"] = pd.cut(
        df["Priority_Score"],
        bins=[-0.01, 40, 70, 100],
        labels=[
            "Low Priority",
            "Medium Priority",
            "High Priority",
        ],
        include_lowest=True,
    ).astype(str)

    df = df.sort_values(
        "Priority_Score", ascending=False
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("CUSTOMER PRIORITY SCORING")
    print("-" * 45)
    print(f"Customers scored: {len(df):,}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nPriority category counts:")
    print(
        df["Priority_Category"]
        .value_counts()
        .reindex(
            ["High Priority", "Medium Priority", "Low Priority"],
            fill_value=0,
        )
        .to_string()
    )

    print("\nTop 10 customers by priority:")
    print(
        df[
            [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "Priority_Score",
                "Priority_Category",
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nImportant: This is a rule-based prioritization "
        "score, not a validated conversion prediction."
    )


if __name__ == "__main__":
    main()
