
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "sales_analysis.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "churn_model_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR / "churn_model_metrics.csv"
)


# ============================================================
# 2. TIME-BASED MODELING PERIODS
# ============================================================

FEATURE_START = pd.Timestamp("2010-12-01")

FEATURE_END = pd.Timestamp(
    "2011-09-30 23:59:59"
)

LABEL_START = pd.Timestamp("2011-10-01")

LABEL_END = pd.Timestamp(
    "2011-12-09 23:59:59"
)


# ============================================================
# 3. LOAD AND CLEAN TRANSACTION DATA
# ============================================================

def load_transaction_data():
    """Load and clean historical retail transactions."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        dtype={"InvoiceNo": str},
    )

    required_columns = {
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
        "Quantity",
        "Revenue",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    for column in [
        "CustomerID",
        "Quantity",
        "Revenue",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
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

    # Keep valid positive sales transactions only.
    df = df[
        (df["Quantity"] > 0)
        & (df["Revenue"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid sales transactions remain after cleaning."
        )

    print(f"Valid transaction rows: {len(df):,}")
    print(
        "Observation period: "
        f"{df['InvoiceDate'].min()} to "
        f"{df['InvoiceDate'].max()}"
    )

    return df


# ============================================================
# 4. CREATE CUSTOMER FEATURES
# ============================================================

def create_customer_features(df):
    """
    Calculate customer features using transactions
    from the feature period only.
    """

    feature_df = df[
        (df["InvoiceDate"] >= FEATURE_START)
        & (df["InvoiceDate"] <= FEATURE_END)
    ].copy()

    if feature_df.empty:
        raise ValueError(
            "No transactions found in the feature period."
        )

    customer_features = (
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

    # Calculate recency relative to the feature cutoff.
    customer_features["Recency_Days"] = (
        FEATURE_END
        - customer_features["Last_Purchase"]
    ).dt.total_seconds() / (24 * 60 * 60)

    customer_features["Average_Order_Value"] = (
        customer_features["Total_Revenue"]
        / customer_features["Total_Invoices"]
    )

    return customer_features


# ============================================================
# 5. CREATE FUTURE-PERIOD CHURN LABELS
# ============================================================

def create_churn_labels(df):
    """
    Churn = 1: customer made no purchase during the
    future label window.

    Churn = 0: customer made at least one purchase
    during the future label window.
    """

    label_df = df[
        (df["InvoiceDate"] >= LABEL_START)
        & (df["InvoiceDate"] <= LABEL_END)
    ]

    future_customers = set(
        label_df["CustomerID"].unique()
    )

    # Use customers with feature-period purchases
    # as the population to be labeled.
    feature_customers = (
        df.loc[
            (df["InvoiceDate"] >= FEATURE_START)
            & (df["InvoiceDate"] <= FEATURE_END),
            "CustomerID",
        ]
        .drop_duplicates()
    )

    churn_labels = pd.DataFrame(
        {"CustomerID": feature_customers}
    )

    churn_labels["Churn"] = (
        ~churn_labels["CustomerID"].isin(
            future_customers
        )
    ).astype(int)

    return churn_labels


# ============================================================
# 6. PREPARE MODEL DATASET
# ============================================================

def prepare_model_dataset(df):
    """Combine historical features and future labels."""

    customer_features = create_customer_features(df)

    churn_labels = create_churn_labels(df)

    model_df = customer_features.merge(
        churn_labels,
        on="CustomerID",
        how="inner",
        validate="one_to_one",
    )

    feature_columns = [
        "Recency_Days",
        "Total_Invoices",
        "Total_Revenue",
        "Total_Transactions",
        "Average_Quantity",
        "Average_Order_Value",
    ]

    if model_df.empty:
        raise ValueError(
            "The model dataset is empty."
        )

    # Features must be numeric and finite.
    model_df[feature_columns] = (
        model_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    model_df = model_df.dropna(
        subset=feature_columns + ["Churn"]
    ).copy()

    if model_df.empty:
        raise ValueError(
            "No customers remain after feature validation."
        )

    if not set(model_df["Churn"].unique()).issubset({0, 1}):
        raise ValueError(
            "Churn labels must contain only 0 and 1."
        )

    if model_df["Churn"].nunique() != 2:
        raise ValueError(
            "Both churn classes are required for evaluation."
        )

    if not np.isfinite(
        model_df[feature_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values remain in model features."
        )

    print(
        f"Customers with feature-period purchases: "
        f"{len(model_df):,}"
    )

    print("\nOverall churn distribution:")
    print(
        model_df["Churn"]
        .value_counts()
        .sort_index()
    )

    return model_df, feature_columns


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

def split_model_data(model_df, feature_columns):
    """Create a reproducible stratified 80/20 split."""

    X = model_df[feature_columns]
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with training-only scaling."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate predictions on the held-out test dataset."""

    y_pred = model.predict(X_test)

    # Probability of the positive class, Churn = 1.
    class_positions = list(model.classes_)
    positive_position = class_positions.index(1)

    y_prob = model.predict_proba(X_test)[
        :, positive_position
    ]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 60)

    print(
        "Positive class: 1 = No purchase in the "
        "future label window"
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        "Rows = actual classes; columns = predicted classes"
    )
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Purchased in label window (0)",
                "No purchase in label window (1)",
            ],
            zero_division=0,
        )
    )

    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
    }

    return results, y_pred, y_prob


# ============================================================
# 11. SAVE CUSTOMER-LEVEL PREDICTIONS
# ============================================================

def save_predictions(
    model_df,
    X_test,
    y_test,
    logistic_pred,
    logistic_prob,
    rf_pred,
    rf_prob,
):
    """Save test-set labels and predictions to a CSV."""

    predictions = pd.DataFrame(
        {
            "CustomerID": model_df.loc[
                X_test.index, "CustomerID"
            ].to_numpy(),
            "Actual_Churn": y_test.to_numpy(),
            "Logistic_Predicted_Churn": logistic_pred,
            "Logistic_Churn_Probability": logistic_prob,
            "RandomForest_Predicted_Churn": rf_pred,
            "RandomForest_Churn_Probability": rf_prob,
        }
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(f"\nPredictions saved to:\n{PREDICTIONS_FILE}")

    return predictions


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():
    """Run the complete churn modeling workflow."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nINTELLIGENT SALES & CUSTOMER ANALYTICS")
    print("CHURN PREDICTION MODEL")
    print("-" * 60)

    print(f"Feature period: {FEATURE_START} to {FEATURE_END}")
    print(f"Label period:   {LABEL_START} to {LABEL_END}")

    df = load_transaction_data()

    model_df, feature_columns = prepare_model_dataset(df)

    X_train, X_test, y_train, y_test = split_model_data(
        model_df,
        feature_columns,
    )

    print("\nMODEL DATASET")
    print("-" * 40)
    print(f"Total customers:   {len(model_df):,}")
    print(f"Training customers: {len(X_train):,}")
    print(f"Testing customers:  {len(X_test):,}")

    print("\nTraining churn distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting churn distribution:")
    print(y_test.value_counts().sort_index())

    print("\nChecking feature values...")

    if not np.isfinite(
        X_train.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Training features contain non-finite values."
        )

    if not np.isfinite(
        X_test.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Testing features contain non-finite values."
        )

    # Train both models.
    print("\nTraining Logistic Regression...")
    logistic_model = train_logistic_regression(
        X_train,
        y_train,
    )

    print("Training Random Forest...")
    random_forest_model = train_random_forest(
        X_train,
        y_train,
    )

    # Evaluate both models on the same test data.
    logistic_results, logistic_pred, logistic_prob = (
        evaluate_model(
            logistic_model,
            X_test,
            y_test,
            "Logistic Regression",
        )
    )

    rf_results, rf_pred, rf_prob = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest",
    )

    # Compare models.
    results_df = pd.DataFrame(
        [logistic_results, rf_results]
    )

    print("\nMODEL COMPARISON")
    print("-" * 75)
    print(results_df.round(4).to_string(index=False))

    # Save metrics.
    results_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(f"\nMetrics saved to:\n{METRICS_FILE}")

    # Save held-out test predictions.
    save_predictions(
        model_df,
        X_test,
        y_test,
        logistic_pred,
        logistic_prob,
        rf_pred,
        rf_prob,
    )

    print("\nWORKFLOW COMPLETED")
    print(
        "Reminder: these predictions concern the "
        "historical label window, not current customers."
    )


if __name__ == "__main__":
    main()
