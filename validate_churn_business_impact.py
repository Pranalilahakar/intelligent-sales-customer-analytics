
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CUSTOMER_FILE = (
    PROCESSED_DIR / "customer_churn_analysis.csv"
)
SUMMARY_FILE = (
    PROCESSED_DIR / "churn_business_impact.csv"
)


def main():
    for file_path in [CUSTOMER_FILE, SUMMARY_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    customers = pd.read_csv(CUSTOMER_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    required_customer_columns = {
        "CustomerID",
        "Churn_Risk_Status",
        "Total_Revenue",
    }
    required_summary_columns = {
        "Churn_Risk_Status",
        "Customers",
        "Total_Revenue",
        "Customer_Percentage",
    }

    missing_customer = (
        required_customer_columns - set(customers.columns)
    )
    missing_summary = (
        required_summary_columns - set(summary.columns)
    )

    if missing_customer or missing_summary:
        raise ValueError(
            f"Customer file missing: {sorted(missing_customer)}; "
            f"Summary file missing: {sorted(missing_summary)}"
        )

    if customers["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customers found.")

    actual = (
        customers.groupby("Churn_Risk_Status")
        .agg(
            Customers=("CustomerID", "nunique"),
            Total_Revenue=("Total_Revenue", "sum"),
        )
        .reset_index()
    )

    comparison = actual.merge(
        summary[
            [
                "Churn_Risk_Status",
                "Customers",
                "Total_Revenue",
            ]
        ],
        on="Churn_Risk_Status",
        how="outer",
        suffixes=("_actual", "_summary"),
        indicator=True,
    )

    count_match = (
        comparison["_merge"].eq("both").all()
        and comparison["Customers_actual"].equals(
            comparison["Customers_summary"]
        )
    )

    revenue_match = (
        comparison["_merge"].eq("both").all()
        and (
            comparison["Total_Revenue_actual"]
            - comparison["Total_Revenue_summary"]
        ).abs().le(0.02).all()
    )

    percentage_total = summary["Customer_Percentage"].sum()
    percentage_match = abs(percentage_total - 100) <= 0.1

    print("CHURN BUSINESS IMPACT VALIDATION")
    print("-" * 45)
    print(f"Customer rows: {len(customers):,}")
    print(f"Summary groups: {len(summary):,}")
    print(f"Customer counts match: {count_match}")
    print(f"Revenue totals match: {revenue_match}")
    print(f"Percentages total: {percentage_total:.2f}%")
    print(f"Percentage check: {percentage_match}")

    print("\nRecalculated business impact:")
    print(actual.to_string(index=False))

    print("\nValidation result:")
    if count_match and revenue_match and percentage_match:
        print("PASS - Business impact figures reconcile.")
    else:
        print("REVIEW REQUIRED - Investigate mismatched figures.")


if __name__ == "__main__":
    main()

