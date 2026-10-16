
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "customer_churn_analysis.csv"
)
OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "churn_business_impact.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "CustomerID",
        "Churn_Risk_Status",
        "Total_Revenue",
        "Total_Invoices",
        "Recency_Days",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    summary = (
        df.groupby("Churn_Risk_Status")
        .agg(
            Customers=("CustomerID", "nunique"),
            Total_Revenue=("Total_Revenue", "sum"),
            Average_Revenue_Per_Customer=("Total_Revenue", "mean"),
            Average_Invoices_Per_Customer=("Total_Invoices", "mean"),
            Average_Recency_Days=("Recency_Days", "mean"),
        )
        .reset_index()
    )

    summary["Customer_Percentage"] = (
        summary["Customers"] / df["CustomerID"].nunique() * 100
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("CHURN BUSINESS IMPACT SUMMARY")
    print("-" * 45)
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
