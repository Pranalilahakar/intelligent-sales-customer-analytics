
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "customer_purchase_propensity_backtest.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "customer_purchase_propensity_threshold_analysis.csv"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    actual = df["Actual_Purchased_Future_Window"].astype(int)
    scores = df["Purchase_Propensity_Score"] / 100

    results = []

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        predicted = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Customers_Targeted": int(predicted.sum()),
            "Accuracy": accuracy_score(actual, predicted),
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "F1_Score": f1_score(
                actual, predicted, zero_division=0
            ),
            "True_Positive": int(tp),
            "False_Positive": int(fp),
            "True_Negative": int(tn),
            "False_Negative": int(fn),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nPURCHASE PROPENSITY THRESHOLD ANALYSIS")
    print("=" * 75)
    print(result_df.round(4).to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")
    print(
        "\nThresholds are evaluated on the same historical test cohort. "
        "Choose a final threshold based on business costs and validate "
        "it on separate data before deployment."
    )


if __name__ == "__main__":
    main()

