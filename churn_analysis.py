
from pathlib import Path

import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_analysis.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Business rule: classify customers using 90 days of inactivity.
INACTIVITY_DAYS = 90


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Invoice numbers are identifiers, so read them as strings.
    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        dtype={"InvoiceNo": str},
    )

    required_columns = {
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
        "Revenue",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )
    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"],
        errors="coerce",
    )
    df["Revenue"] = pd.to_numeric(
        df["Revenue"],
        errors="coerce",
    )

    # Keep only rows that can be used for customer-level analysis.
    customer_df = df.dropna(
        subset=["CustomerID", "InvoiceDate", "InvoiceNo", "Revenue"]
    ).copy()

    customer_df = customer_df[
        (customer_df["Revenue"] > 0)
    ].copy()

    if customer_df.empty:
        raise ValueError(
            "No valid customer transactions remain after cleaning."
        )

    # Use the latest observed transaction date as the cutoff.
    cutoff_date = customer_df["InvoiceDate"].max()

    # Calculate customer-level RFM-style features.
    customer_metrics = (
        customer_df.groupby("CustomerID")
        .agg(
            Last_Purchase=("InvoiceDate", "max"),
            Total_Invoices=("InvoiceNo", "nunique"),
            Total_Revenue=("Revenue", "sum"),
            Total_Transactions=("InvoiceNo", "size"),
        )
        .reset_index()
    )

    customer_metrics["Recency_Days"] = (
        cutoff_date - customer_metrics["Last_Purchase"]
    ).dt.days

    # This is a rule-based inactivity proxy, not confirmed churn.
    customer_metrics["Churn_Risk_Status"] = (
        customer_metrics["Recency_Days"] > INACTIVITY_DAYS
    ).map({
        True: "Inactive_90_Days",
        False: "Active_Within_90_Days",
    })

    customer_metrics["Churn_Rule_Days"] = INACTIVITY_DAYS
    customer_metrics["Observation_Cutoff"] = cutoff_date

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "customer_churn_analysis.csv"

    customer_metrics.to_csv(output_file, index=False)

    print("CUSTOMER CHURN ANALYSIS")
    print("-" * 40)
    print(f"Input transaction rows: {len(df):,}")
    print(f"Rows with identifiable customers: {len(customer_df):,}")
    print(f"Customers analyzed: {len(customer_metrics):,}")
    print(f"Observation cutoff: {cutoff_date}")
    print(f"Inactivity threshold: {INACTIVITY_DAYS} days")
    print("\nCustomer status counts:")
    print(
        customer_metrics["Churn_Risk_Status"]
        .value_counts()
        .to_string()
    )
    print(f"\nOutput saved to: {output_file}")
    print(
        "\nLimitation: this is a retrospective inactivity-based "
        "proxy, not a confirmed churn prediction model."
    )


if __name__ == "__main__":
    main()
