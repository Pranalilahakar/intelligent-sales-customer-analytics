
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_FILE = (
    BASE_DIR / "models"
    / "customer_purchase_propensity_model.joblib"
)

DATASET_FILE = (
    DATA_DIR / "customer_purchase_propensity_dataset.csv"
)
METRICS_FILE = (
    DATA_DIR / "customer_purchase_propensity_metrics.csv"
)
PREDICTIONS_FILE = (
    DATA_DIR / "customer_purchase_propensity_test_predictions.csv"
)


def main():
    print("\nPHASE 6: MODEL OUTPUT VALIDATION")
    print("=" * 55)

    for path in [
        DATASET_FILE,
        METRICS_FILE,
        PREDICTIONS_FILE,
        MODEL_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required output is missing: {path}"
            )
        print(f"FOUND: {path.name}")

    dataset = pd.read_csv(DATASET_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required_features = [
        "Recency_Days",
        "Total_Invoices",
        "Total_Revenue",
        "Total_Transactions",
        "Average_Quantity",
        "Average_Order_Value",
    ]

    required_dataset_columns = (
        ["CustomerID", "Purchased_Future_Window"]
        + required_features
    )

    missing = set(required_dataset_columns) - set(dataset.columns)
    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    if dataset["CustomerID"].isna().any():
        raise ValueError("Missing customer IDs in dataset.")

    if dataset["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customer rows in dataset.")

    if not dataset["Purchased_Future_Window"].isin([0, 1]).all():
        raise ValueError("Target contains values other than 0 and 1.")

    if dataset["Purchased_Future_Window"].nunique() != 2:
        raise ValueError("Both target classes must be present.")

    values = dataset[required_features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinite values.")

    print("\nDATASET CHECKS")
    print(f"Customer rows: {len(dataset):,}")
    print(f"Unique customers: {dataset['CustomerID'].nunique():,}")
    print("Target distribution:")
    print(dataset["Purchased_Future_Window"].value_counts().sort_index())

    required_prediction_columns = [
        "CustomerID",
        "Actual_Purchased_Future_Window",
        "Predicted_Purchased_Future_Window",
        "Purchase_Propensity_Score",
        "Selected_Model",
    ]

    missing = (
        set(required_prediction_columns) - set(predictions.columns)
    )
    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    if predictions["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customers in test predictions.")

    score = predictions["Purchase_Propensity_Score"]

    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("Prediction scores contain invalid values.")

    if not score.between(0, 100).all():
        raise ValueError("Scores must be between 0 and 100.")

    if not predictions[
        "Actual_Purchased_Future_Window"
    ].isin([0, 1]).all():
        raise ValueError("Actual test labels are invalid.")

    if not predictions[
        "Predicted_Purchased_Future_Window"
    ].isin([0, 1]).all():
        raise ValueError("Predicted test labels are invalid.")

    print("\nPREDICTION CHECKS")
    print(f"Test predictions: {len(predictions):,}")
    print(f"Minimum score: {score.min():.2f}")
    print(f"Maximum score: {score.max():.2f}")
    print("Prediction scores are finite and within 0–100.")

    required_metrics = {
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC",
        "Brier_Score",
    }

    missing = required_metrics - set(metrics.columns)
    if missing:
        raise ValueError(
            f"Missing metric columns: {sorted(missing)}"
        )

    numeric_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC",
        "Brier_Score",
    ]

    metric_values = metrics[numeric_metrics].to_numpy(dtype=float)

    if not np.isfinite(metric_values).all():
        raise ValueError("Metrics contain invalid numeric values.")

    for column in numeric_metrics:
        if not metrics[column].between(0, 1).all():
            raise ValueError(
                f"Unexpected metric values in {column}."
            )

    print("\nMODEL METRICS")
    print(metrics.round(4).to_string(index=False))

    artifact = joblib.load(MODEL_FILE)

    for key in [
        "model",
        "feature_columns",
        "selected_model",
        "target_definition",
    ]:
        if key not in artifact:
            raise ValueError(
                f"Model artifact is missing key: {key}"
            )

    if artifact["feature_columns"] != required_features:
        raise ValueError(
            "Saved model feature list differs from expected features."
        )

    print("\nMODEL ARTIFACT CHECK")
    print(f"Selected model: {artifact['selected_model']}")
    print(f"Features: {artifact['feature_columns']}")
    print("Saved model loaded successfully.")

    print("\nPASS: Basic Phase 6 artifact checks completed.")
    print(
        "Note: These checks do not prove temporal generalization, "
        "probability calibration, or production readiness."
    )


if __name__ == "__main__":
    main()
