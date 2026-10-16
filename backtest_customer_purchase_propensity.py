
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_analysis.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "customer_purchase_propensity_backtest.csv"
)
METRICS_FILE = (
    OUTPUT_DIR / "customer_purchase_propensity_backtest_metrics.csv"
)

FEATURES = [
    "Recency_Days",
    "Total_Invoices",
    "Total_Revenue",
    "Total_Transactions",
    "Average_Quantity",
    "Average_Order_Value",
]


def load_transactions():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"InvoiceNo": str},
        low_memory=False,
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    for col in ["CustomerID", "Quantity", "Revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "InvoiceNo", "InvoiceDate",
            "CustomerID", "Quantity", "Revenue",
        ]
    )

    # Match the existing project’s positive-sales definition.
    df = df[
        (df["Quantity"] > 0)
        & (df["Revenue"] > 0)
    ].copy()

    return df


def build_snapshot(df, feature_cutoff, outcome_start, outcome_end):
    """Build features before a cutoff and labels after that cutoff."""

    feature_data = df[
        df["InvoiceDate"] <= pd.Timestamp(feature_cutoff)
    ].copy()

    outcome_data = df[
        df["InvoiceDate"].between(
            pd.Timestamp(outcome_start),
            pd.Timestamp(outcome_end) + pd.Timedelta(days=1)
            - pd.Timedelta(microseconds=1),
        )
    ]

    if feature_data.empty or outcome_data.empty:
        raise ValueError(
            f"Insufficient data for cutoff {feature_cutoff} "
            f"and outcome {outcome_start} to {outcome_end}."
        )

    cutoff = pd.Timestamp(feature_cutoff) + pd.Timedelta(days=1)

    features = (
        feature_data.groupby("CustomerID")
        .agg(
            Last_Purchase=("InvoiceDate", "max"),
            Total_Invoices=("InvoiceNo", "nunique"),
            Total_Revenue=("Revenue", "sum"),
            Total_Transactions=("InvoiceNo", "size"),
            Average_Quantity=("Quantity", "mean"),
        )
        .reset_index()
    )

    features["Recency_Days"] = (
        cutoff - features["Last_Purchase"]
    ).dt.total_seconds() / 86400

    features["Average_Order_Value"] = (
        features["Total_Revenue"] / features["Total_Invoices"]
    )

    future_customers = set(outcome_data["CustomerID"].unique())

    features["Purchased_Future_Window"] = (
        features["CustomerID"].isin(future_customers).astype(int)
    )

    features[FEATURES] = (
        features[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    features = features.dropna(
        subset=FEATURES + ["Purchased_Future_Window"]
    ).copy()

    return features


def evaluate(model_name, y_true, predictions, probabilities):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "Brier_Score": brier_score_loss(y_true, probabilities),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCHRONOLOGICAL PURCHASE PROPENSITY BACKTEST")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = load_transactions()

    print(f"Valid transaction rows: {len(df):,}")
    print(f"Data range: {df['InvoiceDate'].min()} to "
          f"{df['InvoiceDate'].max()}")

    # Earlier cohort: features through June, outcome July-August.
    train_df = build_snapshot(
        df,
        feature_cutoff="2011-06-30 23:59:59",
        outcome_start="2011-07-01",
        outcome_end="2011-08-31",
    )

    # Later cohort: features through August, outcome September-October.
    test_df = build_snapshot(
        df,
        feature_cutoff="2011-08-31 23:59:59",
        outcome_start="2011-09-01",
        outcome_end="2011-10-31",
    )

    X_train = train_df[FEATURES]
    y_train = train_df["Purchased_Future_Window"]

    X_test = test_df[FEATURES]
    y_test = test_df["Purchased_Future_Window"]

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Both outcome classes must exist in training and test cohorts."
        )

    print("\nTEMPORAL COHORTS")
    print(f"Training customers: {len(train_df):,}")
    print(f"Training outcome distribution:\n{y_train.value_counts().sort_index()}")
    print(f"\nTest customers: {len(test_df):,}")
    print(f"Test outcome distribution:\n{y_test.value_counts().sort_index()}")

    # All preprocessing is fitted on training data only.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
        )),
    ])

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)

    model_probability = model.predict_proba(X_test)[:, 1]
    model_prediction = (model_probability >= 0.5).astype(int)

    baseline_probability = baseline.predict_proba(X_test)[:, 1]
    baseline_prediction = baseline.predict(X_test)

    results = pd.DataFrame([
        evaluate(
            "Training-period prevalence baseline",
            y_test,
            baseline_prediction,
            baseline_probability,
        ),
        evaluate(
            "Logistic Regression - chronological test",
            y_test,
            model_prediction,
            model_probability,
        ),
    ])

    print("\nCHRONOLOGICAL TEST RESULTS")
    print(results.round(4).to_string(index=False))

    results.to_csv(METRICS_FILE, index=False)

    predictions = pd.DataFrame({
        "CustomerID": test_df["CustomerID"].to_numpy(),
        "Actual_Purchased_Future_Window": y_test.to_numpy(),
        "Predicted_Purchased_Future_Window": model_prediction,
        "Purchase_Propensity_Score": (
            model_probability * 100
        ).round(2),
        "Feature_Cutoff": "2011-08-31",
        "Outcome_Window": "2011-09-01 to 2011-10-31",
    })

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    print(f"\nMetrics saved: {METRICS_FILE}")
    print(f"Predictions saved: {PREDICTIONS_FILE}")
    print("\nBACKTEST COMPLETED")
    print(
        "These results measure performance on a later historical cohort; "
        "they do not establish performance on current customers."
    )


if __name__ == "__main__":
    main()
