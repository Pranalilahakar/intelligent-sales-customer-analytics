from pathlib import Path

import pandas as pd

BASE = Path("data/processed")
PRIORITY_FILE = BASE / "customer_priority_scores.csv"
CHURN_FILE = BASE / "customer_churn_analysis.csv"
OUTPUT_FILE = BASE / "customer_action_recommendations.csv"


def recommend_action(row, monetary_threshold, frequency_threshold):
    status = row["Churn_Risk_Status"]
    segment = row["Segment_Name"]
    score = row["Priority_Score"]

    if status == "Inactive_90_Days":
        if (
            row["Monetary"] >= monetary_threshold
            and row["Frequency"] >= frequency_threshold
        ):
            return pd.Series({
                "Action_Priority": "High",
                "Recommended_Action": "High-Value Win-Back Campaign",
                "Recommendation_Reason": (
                    "Inactive customer with strong historical value or priority score."
                ),
            })

        return pd.Series({
            "Action_Priority": "Medium",
            "Recommended_Action": "Customer Reactivation Campaign",
            "Recommendation_Reason": (
                "Customer has had no purchase within the defined 90-day inactivity rule."
            ),
        })

    if status == "Active_Within_90_Days":
        if score >= 70:
            return pd.Series({
                "Action_Priority": "Medium",
                "Recommended_Action": "Loyalty and Upsell Campaign",
                "Recommendation_Reason": (
                    "Active customer with a high existing priority score."
                ),
            })

        if segment == "Recent Low-Spend":
            return pd.Series({
                "Action_Priority": "Medium",
                "Recommended_Action": "Second-Purchase Nurture Campaign",
                "Recommendation_Reason": (
                    "Recently active customer with relatively low historical spending."
                ),
            })

        if segment == "Regular Customers":
            return pd.Series({
                "Action_Priority": "Medium",
                "Recommended_Action": "Repeat-Purchase Campaign",
                "Recommendation_Reason": (
                    "Active regular customer who may respond to relevant repeat-purchase offers."
                ),
            })

        return pd.Series({
            "Action_Priority": "Low",
            "Recommended_Action": "Low-Cost Customer Engagement",
            "Recommendation_Reason": (
                "Active customer suitable for low-cost engagement and monitoring."
            ),
        })

    raise ValueError(f"Unexpected churn status: {status!r}")


def main():
    priority = pd.read_csv(PRIORITY_FILE)
    churn = pd.read_csv(CHURN_FILE)

    required_priority = {
        "CustomerID", "Segment_Name", "Priority_Score", "Priority_Category"
    }
    required_churn = {
        "CustomerID", "Churn_Risk_Status", "Recency_Days"
    }

    missing_priority = required_priority - set(priority.columns)
    missing_churn = required_churn - set(churn.columns)

    if missing_priority or missing_churn:
        raise ValueError(
            f"Missing priority columns: {sorted(missing_priority)}; "
            f"missing churn columns: {sorted(missing_churn)}"
        )

    if priority["CustomerID"].isna().any() or churn["CustomerID"].isna().any():
        raise ValueError("CustomerID contains missing values.")

    if priority["CustomerID"].duplicated().any():
        raise ValueError("Priority data contains duplicate CustomerID values.")

    if churn["CustomerID"].duplicated().any():
        raise ValueError("Churn data contains duplicate CustomerID values.")

    priority["Priority_Score"] = pd.to_numeric(
        priority["Priority_Score"], errors="raise"
    )
    churn["Recency_Days"] = pd.to_numeric(
        churn["Recency_Days"], errors="raise"
    )

    customers = priority.merge(
        churn[
            ["CustomerID", "Churn_Risk_Status", "Recency_Days"]
        ],
        on="CustomerID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = customers["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} priority customers have no matching churn record."
        )

    customers = customers.drop(columns="_merge")

    monetary_threshold = customers["Monetary"].quantile(0.75)
    frequency_threshold = customers["Frequency"].quantile(0.75)

    recommendations = customers.apply(
        lambda row: recommend_action(
            row, monetary_threshold, frequency_threshold
        ),
        axis=1,
    )
    result = pd.concat([customers, recommendations], axis=1)

    result = result.sort_values(
        ["Action_Priority", "Priority_Score"],
        ascending=[True, False],
        key=lambda column: (
            column.map({"High": 0, "Medium": 1, "Low": 2})
            if column.name == "Action_Priority"
            else column
        ),
    ).reset_index(drop=True)

    if len(result) != len(priority):
        raise ValueError("Customer count changed unexpectedly.")

    if result["CustomerID"].duplicated().any():
        raise ValueError("Output contains duplicate customers.")

    BASE.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("CUSTOMER ACTION RECOMMENDATIONS CREATED")
    print("=" * 60)
    print(f"Customers processed: {len(result)}")
    print(f"Unique customers: {result['CustomerID'].nunique()}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nACTION PRIORITY COUNTS")
    print(result["Action_Priority"].value_counts().to_string())

    print("\nRECOMMENDED ACTION COUNTS")
    print(result["Recommended_Action"].value_counts().to_string())

    print("\nTOP 10 RECOMMENDATIONS")
    print(
        result[
            [
                "CustomerID",
                "Segment_Name",
                "Churn_Risk_Status",
                "Priority_Score",
                "Action_Priority",
                "Recommended_Action",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
