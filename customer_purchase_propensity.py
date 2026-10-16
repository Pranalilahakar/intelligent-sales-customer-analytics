
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "sales_analysis.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

DATASET_FILE = OUTPUT_DIR / "customer_purchase_propensity_dataset.csv"
METRICS_FILE = OUTPUT_DIR / "customer_purchase_propensity_metrics.csv"
PREDICTIONS_FILE = (
    OUTPUT_DIR / "customer_purchase_propensity_test_predictions.csv"
)
MODEL_FILE = MODEL_DIR / "customer_purchase_propensity_model.joblib"


# ============================================================
# 2. FEATURE AND OUTCOME WINDOWS
# ============================================================

FEATURE_START = pd.Timestamp("2010-12-01")
FEATURE_END = pd.Timestamp("2011-09-30 23:59:59")
LABEL_START = pd.Timestamp("2011-10-01")


# ============================================================
# 3. LOAD AND VALIDATE TRANSACTIONS
# ============================================================

def load_transactions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"InvoiceNo": str},
        low_memory=False,
    )

    required = {
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
        "Quantity",
        "Revenue",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    for column in ["CustomerID", "Quantity", "Revenue"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "InvoiceNo",
            "InvoiceDate",
            "CustomerID",
            "Quantity",
            "Revenue",
        ]
    ).copy()

    # Follow the same positive-sales convention as the existing
    # churn model. Returns and cancellations are not modeled here.
    df = df[
        (df["Quantity"] > 0)
        & (df["Revenue"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid positive sales remain.")

    observed_end = df["InvoiceDate"].max()

    if observed_end < LABEL_START:
        raise ValueError(
            "No transactions exist in the outcome window."
        )

    print("DATA COVERAGE")
    print(f"First transaction: {df['InvoiceDate'].min()}")
    print(f"Last transaction:  {observed_end}")
    print(f"Valid transaction rows: {len(df):,}")

    return df, observed_end


# ============================================================
# 4. BUILD CUSTOMER FEATURES AND TARGET
# ============================================================

def build_dataset(df, observed_end):
    feature_df = df[
        df["InvoiceDate"].between(
            FEATURE_START, FEATURE_END
        )
    ].copy()

    outcome_df = df[
        (df["InvoiceDate"] >= LABEL_START)
        & (df["InvoiceDate"] <= observed_end)
    ].copy()

    if feature_df.empty or outcome_df.empty:
        raise ValueError(
            "Feature or outcome window contains no transactions."
        )

    features = (
        feature_df.groupby("CustomerID")
        .agg(
            Last_Purchase=("InvoiceDate", "max"),
            Total_Invoices=("InvoiceNo", "nunique"),
            Total_Revenue=("Revenue", "sum"),
            Total_Transactions=("InvoiceNo", "size"),
            Average_Quantity=("Quantity", "mean"),
        )
        .reset_index()
    )

    # Calculate recency only using historical feature-window data.
    features["Recency_Days"] = (
        FEATURE_END - features["Last_Purchase"]
    ).dt.total_seconds() / 86400

    features["Average_Order_Value"] = (
        features["Total_Revenue"]
        / features["Total_Invoices"]
    )

    # Target: 1 = at least one future purchase;
    #         0 = no purchase observed in the outcome window.
    future_customers = set(outcome_df["CustomerID"].unique())

    features["Purchased_Future_Window"] = (
        features["CustomerID"]
        .isin(future_customers)
        .astype(int)
    )

    feature_columns = [
        "Recency_Days",
        "Total_Invoices",
        "Total_Revenue",
        "Total_Transactions",
        "Average_Quantity",
        "Average_Order_Value",
    ]

    features = features.replace(
        [np.inf, -np.inf], np.nan
    )

    features = features.dropna(
        subset=feature_columns + ["Purchased_Future_Window"]
    ).copy()

    if features["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customer records found.")

    if features["Purchased_Future_Window"].nunique() != 2:
        raise ValueError("Both target classes are required.")

    features = features.reset_index(drop=True)

    print("\nTARGET DEFINITION")
    print(f"Feature window: {FEATURE_START} to {FEATURE_END}")
    print(f"Outcome window: {LABEL_START} to {observed_end}")
    print("1 = purchase observed in outcome window")
    print("0 = no purchase observed in outcome window")

    print("\nTARGET DISTRIBUTION")
    print(
        features["Purchased_Future_Window"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    features.to_csv(DATASET_FILE, index=False)
    print(f"\nDataset saved: {DATASET_FILE}")

    return features, feature_columns


# ============================================================
# 5. DEFINE MODELS
# ============================================================

def create_models():
    return {
        "Baseline": DummyClassifier(strategy="prior"),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ============================================================
# 6. EVALUATE MODELS
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(
            y_test, probabilities
        ),
        "Brier_Score": brier_score_loss(
            y_test, probabilities
        ),
    }


# ============================================================
# 7. TRAIN, COMPARE, AND SAVE
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCUSTOMER PURCHASE PROPENSITY")
    print("=" * 55)

    transactions, observed_end = load_transactions()

    data, feature_columns = build_dataset(
        transactions, observed_end
    )

    X = data[feature_columns]
    y = data["Purchased_Future_Window"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nDATASET SPLIT")
    print(f"Total customers: {len(data):,}")
    print(f"Training customers: {len(X_train):,}")
    print(f"Testing customers: {len(X_test):,}")

    models = create_models()
    results = []
    fitted_models = {}

    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        estimator.fit(X_train, y_train)
        fitted_models[name] = estimator

        results.append(
            evaluate_model(
                name, estimator, X_test, y_test
            )
        )

    metrics = pd.DataFrame(results)

    # The baseline is a reference, not a candidate for deployment.
    candidates = metrics[
        metrics["Model"] != "Baseline"
    ].sort_values(
        ["PR_AUC", "ROC_AUC"],
        ascending=False,
    )

    selected_name = candidates.iloc[0]["Model"]

    metrics["Selected_For_Final_Model"] = (
        metrics["Model"] == selected_name
    )

    metrics.to_csv(METRICS_FILE, index=False)

    print("\nMODEL COMPARISON")
    print(metrics.round(4).to_string(index=False))
    print(f"\nMetrics saved: {METRICS_FILE}")

    # Save held-out predictions from the selected model.
    selected_test_model = fitted_models[selected_name]

    probabilities = selected_test_model.predict_proba(X_test)[
        :, list(selected_test_model.classes_).index(1)
    ]

    test_predictions = pd.DataFrame({
        "CustomerID": data.loc[X_test.index, "CustomerID"].to_numpy(),
        "Actual_Purchased_Future_Window": y_test.to_numpy(),
        "Predicted_Purchased_Future_Window": (
            selected_test_model.predict(X_test)
        ),
        "Purchase_Propensity_Score": (
            probabilities * 100
        ).round(2),
        "Selected_Model": selected_name,
    })

    test_predictions.to_csv(
        PREDICTIONS_FILE, index=False
    )

    print(f"Test predictions saved: {PREDICTIONS_FILE}")

    # Refit the selected model on all labeled data for later use.
    # Its held-out metrics above remain the evaluation results.
    final_model = clone(models[selected_name])
    final_model.fit(X, y)

    joblib.dump({
        "model": final_model,
        "feature_columns": feature_columns,
        "selected_model": selected_name,
        "target_column": "Purchased_Future_Window",
        "target_definition": (
            "At least one qualifying positive sales transaction "
            "during the observed outcome window."
        ),
        "feature_start": str(FEATURE_START),
        "feature_end": str(FEATURE_END),
        "label_start": str(LABEL_START),
        "label_end": str(observed_end),
    }, MODEL_FILE)

    print(f"Final model saved: {MODEL_FILE}")
    print(f"Selected model: {selected_name}")

    print("\nPHASE 6 INITIAL MODELING COMPLETE")
    print(
        "Note: held-out evaluation uses a random customer split. "
        "This is not yet a chronological backtest or proof of "
        "performance on future business periods."
    )
    print(
        "The propensity score is not a lead-conversion probability "
        "and may not be calibrated."
    )


if __name__ == "__main__":
    main()
