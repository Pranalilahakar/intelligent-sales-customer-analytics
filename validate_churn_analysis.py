
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "customer_churn_analysis.csv"
)

INACTIVITY_DAYS = 90


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Churn analysis file not found: {INPUT_FILE}"
        )

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
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["Recency_Days"] = pd.to_numeric(
        df["Recency_Days"], errors="coerce"
    )
    df["Total_Revenue"] = pd.to_numeric(
        df["Total_Revenue"], errors="coerce"
    )
    df["Total_Invoices"] = pd.to_numeric(
        df["Total_Invoices"], errors="coerce"
    )

    duplicate_customers = df["CustomerID"].duplicated().sum()
    missing_customer_ids = df["CustomerID"].isna().sum()

    expected_status = df["Recency_Days"].gt(
        INACTIVITY_DAYS
    ).map({
        True: "Inactive_90_Days",
        False: "Active_Within_90_Days",
    })

    status_mismatches = (
        df["Churn_Risk_Status"] != expected_status
    ).sum()

    invalid_recency = (
        df["Recency_Days"].isna()
        | (df["Recency_Days"] < 0)
    ).sum()

    invalid_revenue = (
        df["Total_Revenue"].isna()
        | (df["Total_Revenue"] < 0)
    ).sum()

    invalid_invoices = (
        df["Total_Invoices"].isna()
        | (df["Total_Invoices"] <= 0)
    ).sum()

    print("CHURN ANALYSIS VALIDATION")
    print("-" * 45)
    print(f"Customer records: {len(df):,}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"Duplicate customer rows: {duplicate_customers:,}")
    print(f"Missing CustomerIDs: {missing_customer_ids:,}")
    print(f"Status mismatches: {status_mismatches:,}")
    print(f"Invalid recency values: {invalid_recency:,}")
    print(f"Invalid revenue values: {invalid_revenue:,}")
    print(f"Invalid invoice values: {invalid_invoices:,}")

    print("\nCustomer status counts:")
    print(df["Churn_Risk_Status"].value_counts().to_string())

    checks_passed = (
        duplicate_customers == 0
        and missing_customer_ids == 0
        and status_mismatches == 0
        and invalid_recency == 0
        and invalid_revenue == 0
        and invalid_invoices == 0
    )

    print("\nValidation result:")
    if checks_passed:
        print("PASS - All defined data-quality checks passed.")
    else:
        print(
            "REVIEW REQUIRED - One or more checks failed."
        )

    print(
        "\nNote: Passing these checks does not validate "
        "churn prediction accuracy."
    )


if __name__ == "__main__":
    main()
