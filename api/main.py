
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SALES_FILE = PROCESSED_DIR / "sales_analysis.csv"
SEGMENTS_FILE = PROCESSED_DIR / "customer_segments.csv"
SEGMENT_SUMMARY_FILE = PROCESSED_DIR / "customer_segment_summary.csv"
PRIORITY_FILE = PROCESSED_DIR / "customer_priority_scores.csv"
CUSTOMER_ACTIONS_FILE = (
    PROCESSED_DIR / "customer_action_recommendations.csv"
)

CHURN_PREDICTIONS_FILE = (
    PROCESSED_DIR / "churn_model_predictions.csv"
)

PURCHASE_PROPENSITY_MODEL_FILE = (
    BASE_DIR / "models" / "customer_purchase_propensity_model.joblib"
)

PURCHASE_PROPENSITY_PREDICTIONS_FILE = (
    PROCESSED_DIR / "customer_purchase_propensity_test_predictions.csv"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Intelligent Sales & Customer Analytics API",
    description=(
        "API for sales summaries, customer segmentation, "
        "rule-based customer prioritization, historical churn "
        "predictions, and experimental customer purchase propensity scoring."
    ),
    version="1.1.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PurchasePropensityInput(BaseModel):
    """Validate customer features required by the propensity model."""

    model_config = ConfigDict(extra="forbid")

    Recency_Days: float = Field(ge=0, allow_inf_nan=False)
    Total_Invoices: int = Field(ge=0)
    Total_Revenue: float = Field(ge=0, allow_inf_nan=False)
    Total_Transactions: int = Field(ge=0)
    Average_Quantity: float = Field(ge=0, allow_inf_nan=False)
    Average_Order_Value: float = Field(ge=0, allow_inf_nan=False)


# ============================================================
# SHARED HELPER FUNCTIONS
# ============================================================

def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a processed CSV file and report useful errors."""

    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Required data file not found: {file_path.name}",
        )

    try:
        return pd.read_csv(file_path, low_memory=False)

    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {file_path.name}: {exc}",
        ) from exc


def records_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-friendly records."""

    clean_df = df.astype(object).where(pd.notna(df), None)
    return clean_df.to_dict(orient="records")


def customer_id_matches(
    series: pd.Series,
    customer_id: str,
) -> pd.Series:
    """Match customer IDs whether pandas reads them as integers or floats."""

    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        == customer_id
    )


@lru_cache(maxsize=1)
def load_purchase_propensity_model() -> dict:
    """Load and cache the saved purchase propensity model artifact."""

    if not PURCHASE_PROPENSITY_MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Purchase propensity model file not found.",
        )

    try:
        artifact = joblib.load(PURCHASE_PROPENSITY_MODEL_FILE)

    except Exception as exc:
        # Do not expose internal file paths or deserialization details.
        raise HTTPException(
            status_code=500,
            detail="Could not load the purchase propensity model.",
        ) from exc

    required_keys = {
        "model",
        "feature_columns",
        "selected_model",
        "target_column",
    }

    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise HTTPException(
            status_code=500,
            detail="Purchase propensity model artifact is incomplete.",
        )

    if not isinstance(artifact["feature_columns"], list):
        raise HTTPException(
            status_code=500,
            detail="Purchase propensity model features are invalid.",
        )

    return artifact


def validate_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    file_description: str,
) -> None:
    """Raise an API error if a DataFrame lacks required columns."""

    missing = required_columns - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{file_description} is missing columns: "
                f"{sorted(missing)}"
            ),
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """Return API welcome information."""

    return {
        "message": "Welcome to the Intelligent Sales & Customer Analytics API",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    """Check availability of required data files and model artifact."""

    files = {
        "sales_analysis": SALES_FILE,
        "customer_segments": SEGMENTS_FILE,
        "customer_segment_summary": SEGMENT_SUMMARY_FILE,
        "customer_priority_scores": PRIORITY_FILE,
        "customer_action_recommendations": CUSTOMER_ACTIONS_FILE,
        "churn_model_predictions": CHURN_PREDICTIONS_FILE,
        "purchase_propensity_model": PURCHASE_PROPENSITY_MODEL_FILE,
        "purchase_propensity_predictions": PURCHASE_PROPENSITY_PREDICTIONS_FILE,
    }

    file_status = {
        name: path.is_file()
        for name, path in files.items()
    }

    return {
        "status": "healthy" if all(file_status.values()) else "degraded",
        "data_files": file_status,
    }


# ============================================================
# SALES SUMMARY
# ============================================================

@app.get("/summary")
def summary():
    """Return high-level sales and customer statistics."""

    sales = load_csv(SALES_FILE)
    segments = load_csv(SEGMENTS_FILE)
    priority = load_csv(PRIORITY_FILE)

    required_sales_columns = {
        "Revenue",
        "Quantity",
        "InvoiceNo",
    }

    validate_columns(
        sales,
        required_sales_columns,
        "Sales file",
    )

    validate_columns(
        segments,
        {"CustomerID"},
        "Segments file",
    )

    validate_columns(
        priority,
        {"CustomerID"},
        "Priority file",
    )

    revenue = pd.to_numeric(sales["Revenue"], errors="coerce")
    quantity = pd.to_numeric(sales["Quantity"], errors="coerce")

    result = {
        "total_revenue": round(float(revenue.sum()), 2),
        "total_quantity": int(quantity.sum()),
        "unique_invoices": int(sales["InvoiceNo"].nunique()),
        "customers_analyzed": int(segments["CustomerID"].nunique()),
        "customers_scored": int(priority["CustomerID"].nunique()),
    }

    if "Country" in sales.columns:
        result["countries_count"] = int(sales["Country"].nunique())

    return result


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

@app.get("/segments")
def list_segments():
    """Return the customer segment summary."""

    segment_summary = load_csv(SEGMENT_SUMMARY_FILE)

    return {
        "count": len(segment_summary),
        "segments": records_from_dataframe(segment_summary),
    }


@app.get("/segments/{customer_id}")
def get_customer_segment(customer_id: str):
    """Return segment and RFM information for one customer."""

    segments = load_csv(SEGMENTS_FILE)

    validate_columns(
        segments,
        {"CustomerID"},
        "Segments file",
    )

    matches = segments[
        customer_id_matches(segments["CustomerID"], customer_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found.",
        )

    return records_from_dataframe(matches)[0]


# ============================================================
# CUSTOMER PRIORITY
# ============================================================

@app.get("/priority")
def list_customer_priorities(
    category: str | None = Query(
        default=None,
        description=(
            "Optional category: High Priority, "
            "Medium Priority, or Low Priority"
        ),
    ),
    limit: int = Query(default=20, ge=1, le=500),
):
    """Return customers ordered by their rule-based priority score."""

    priority = load_csv(PRIORITY_FILE)

    required_columns = {
        "CustomerID",
        "Priority_Score",
        "Priority_Category",
    }

    validate_columns(priority, required_columns, "Priority file")

    if category is not None:
        valid_categories = {
            "High Priority",
            "Medium Priority",
            "Low Priority",
        }

        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Category must be one of: {sorted(valid_categories)}",
            )

        priority = priority[
            priority["Priority_Category"].astype(str) == category
        ].copy()
    else:
        priority = priority.copy()

    priority["Priority_Score"] = pd.to_numeric(
        priority["Priority_Score"],
        errors="coerce",
    )

    priority = priority.sort_values(
        "Priority_Score",
        ascending=False,
        na_position="last",
    ).head(limit)

    return {
        "count": len(priority),
        "category_filter": category,
        "limit": limit,
        "results": records_from_dataframe(priority),
        "note": (
            "Priority scores are rule-based and are not "
            "validated conversion predictions."
        ),
    }


@app.get("/priority/{customer_id}")
def get_customer_priority(customer_id: str):
    """Return the priority score and details for one customer."""

    priority = load_csv(PRIORITY_FILE)

    validate_columns(
        priority,
        {"CustomerID"},
        "Priority file",
    )

    matches = priority[
        customer_id_matches(priority["CustomerID"], customer_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found.",
        )

    result = records_from_dataframe(matches)[0]

    result["score_interpretation"] = (
        "Rule-based RFM prioritization; not a validated prediction."
    )

    return result


# ============================================================
# HISTORICAL CHURN PREDICTIONS
# ============================================================

@app.get("/churn")
def list_churn_predictions(
    model: str = Query(
        default="logistic",
        description="Model to use: logistic or random_forest",
    ),
    limit: int = Query(default=20, ge=1, le=500),
):
    """Return saved historical churn test predictions."""

    model = model.strip().lower()

    probability_columns = {
        "logistic": "Logistic_Churn_Probability",
        "random_forest": "RandomForest_Churn_Probability",
    }

    prediction_columns = {
        "logistic": "Logistic_Predicted_Churn",
        "random_forest": "RandomForest_Predicted_Churn",
    }

    if model not in probability_columns:
        raise HTTPException(
            status_code=400,
            detail="Model must be 'logistic' or 'random_forest'.",
        )

    predictions = load_csv(CHURN_PREDICTIONS_FILE)

    required_columns = {
        "CustomerID",
        "Actual_Churn",
        probability_columns[model],
        prediction_columns[model],
    }

    validate_columns(
        predictions,
        required_columns,
        "Churn predictions file",
    )

    probability_column = probability_columns[model]

    predictions[probability_column] = pd.to_numeric(
        predictions[probability_column],
        errors="coerce",
    )

    predictions = predictions.dropna(
        subset=[probability_column]
    )

    predictions = predictions[
        predictions[probability_column].between(0, 1)
    ]

    predictions = predictions.sort_values(
        probability_column,
        ascending=False,
    ).head(limit)

    return {
        "count": len(predictions),
        "model": model,
        "limit": limit,
        "results": records_from_dataframe(predictions),
        "note": (
            "These are saved predictions for historical test customers, "
            "not live predictions for all customers."
        ),
    }


@app.get("/churn/{customer_id}")
def get_customer_churn_prediction(customer_id: str):
    """Return saved historical churn predictions for one test customer."""

    predictions = load_csv(CHURN_PREDICTIONS_FILE)

    validate_columns(
        predictions,
        {"CustomerID"},
        "Churn predictions file",
    )

    matches = predictions[
        customer_id_matches(predictions["CustomerID"], customer_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found in the saved churn test predictions.",
        )

    return {
        **records_from_dataframe(matches)[0],
        "note": (
            "Historical test-set prediction. Actual_Churn describes "
            "the historical label window, not current customer status."
        ),
    }


# ============================================================
# HISTORICAL PURCHASE PROPENSITY PREDICTIONS
# ============================================================

@app.get("/purchase-propensity")
def list_purchase_propensity_predictions(
    limit: int = Query(default=20, ge=1, le=500),
):
    """Return saved historical purchase propensity test predictions."""

    predictions = load_csv(PURCHASE_PROPENSITY_PREDICTIONS_FILE)

    required_columns = {
        "CustomerID",
        "Actual_Purchased_Future_Window",
        "Predicted_Purchased_Future_Window",
        "Purchase_Propensity_Score",
        "Selected_Model",
    }

    validate_columns(
        predictions,
        required_columns,
        "Purchase propensity predictions file",
    )

    score_column = "Purchase_Propensity_Score"

    predictions[score_column] = pd.to_numeric(
        predictions[score_column],
        errors="coerce",
    )

    predictions = predictions.dropna(subset=[score_column])

    # Historical scores are stored on a 0–100 scale.
    predictions = predictions[
        predictions[score_column].between(0, 100)
    ]

    predictions = predictions.sort_values(
        score_column,
        ascending=False,
    ).head(limit)

    return {
        "count": len(predictions),
        "limit": limit,
        "score_scale": "0-100",
        "results": records_from_dataframe(predictions),
        "note": (
            "Saved historical test-set predictions, not live predictions "
            "for all customers. Scores are model outputs, not calibrated "
            "probabilities. The source data ends in 2011."
        ),
    }


# ============================================================
# LIVE PURCHASE PROPENSITY SCORING
# ============================================================

@app.post("/purchase-propensity/score")
def score_purchase_propensity(
    customer: PurchasePropensityInput,
):
    """Score supplied customer features using the saved propensity model."""

    artifact = load_purchase_propensity_model()

    feature_columns = artifact["feature_columns"]
    model = artifact["model"]

    input_data = customer.model_dump()

    missing_features = set(feature_columns) - set(input_data)

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model features are missing from the request schema: "
                f"{sorted(missing_features)}"
            ),
        )

    if not hasattr(model, "classes_"):
        raise HTTPException(
            status_code=500,
            detail="The purchase propensity model has not been fitted.",
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise HTTPException(
            status_code=500,
            detail="The model does not contain the positive target class.",
        )

    try:
        features = pd.DataFrame(
            [[input_data[name] for name in feature_columns]],
            columns=feature_columns,
        )

        predicted_class = int(model.predict(features)[0])

        probabilities = model.predict_proba(features)[0]

        positive_class_index = classes.index(1)

        raw_score = float(probabilities[positive_class_index])

        if not 0 <= raw_score <= 1:
            raise ValueError("Model returned an invalid score.")

        # Use the same 0–100 scale as the saved historical scores.
        purchase_score = round(raw_score * 100, 2)

    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The purchase propensity model could not score this input.",
        ) from exc

    return {
        "predicted_purchased_future_window": predicted_class,
        "purchase_propensity_score": purchase_score,
        "score_scale": "0-100",
        "selected_model": artifact["selected_model"],
        "target": artifact["target_column"],
        "note": (
            "Experimental model score on a 0-100 scale. It is not a "
            "validated or calibrated probability for modern customers. "
            "The model was trained on historical retail data ending in 2011."
        ),
    }

# ============================================================
# CUSTOMER ACTION RECOMMENDATIONS
# ============================================================

@app.get("/customer-actions")
def list_customer_actions(
    action_priority: str | None = Query(
        default=None,
        description="Optional action priority: High, Medium, or Low",
    ),
    recommended_action: str | None = Query(
        default=None,
        description="Optional exact recommended action filter",
    ),
    limit: int = Query(default=20, ge=1, le=500),
):
    """Return customer action recommendations."""

    actions = load_csv(CUSTOMER_ACTIONS_FILE)

    required_columns = {
        "CustomerID",
        "Segment_Name",
        "Churn_Risk_Status",
        "Priority_Score",
        "Action_Priority",
        "Recommended_Action",
        "Recommendation_Reason",
    }

    validate_columns(actions, required_columns, "Customer action recommendations file")

    valid_priorities = {"High", "Medium", "Low"}
    if action_priority is not None:
        if action_priority not in valid_priorities:
            raise HTTPException(
                status_code=400,
                detail="Action priority must be High, Medium, or Low.",
            )
        actions = actions[
            actions["Action_Priority"].astype(str) == action_priority
        ].copy()

    if recommended_action is not None:
        valid_actions = {
            "High-Value Win-Back Campaign",
            "Customer Reactivation Campaign",
            "Loyalty and Upsell Campaign",
            "Second-Purchase Nurture Campaign",
            "Repeat-Purchase Campaign",
            "Low-Cost Customer Engagement",
        }

        if recommended_action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail="Unknown recommended action.",
            )

        actions = actions[
            actions["Recommended_Action"].astype(str) == recommended_action
        ].copy()

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    actions["_priority_order"] = actions["Action_Priority"].map(priority_order)
    actions["Priority_Score"] = pd.to_numeric(
        actions["Priority_Score"], errors="coerce"
    )

    actions = actions.sort_values(
        ["_priority_order", "Priority_Score"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns="_priority_order").head(limit)

    return {
        "count": len(actions),
        "action_priority_filter": action_priority,
        "recommended_action_filter": recommended_action,
        "limit": limit,
        "results": records_from_dataframe(actions),
        "note": (
            "Recommendations are based on transparent business rules and "
            "historical retail data. They do not guarantee campaign success."
        ),
    }


@app.get("/customer-actions/{customer_id}")
def get_customer_action(customer_id: str):
    """Return the action recommendation for one customer."""

    actions = load_csv(CUSTOMER_ACTIONS_FILE)

    validate_columns(
        actions,
        {
            "CustomerID",
            "Action_Priority",
            "Recommended_Action",
            "Recommendation_Reason",
        },
        "Customer action recommendations file",
    )

    matches = actions[
        customer_id_matches(actions["CustomerID"], customer_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer action recommendation not found.",
        )

    result = records_from_dataframe(matches)[0]
    result["recommendation_type"] = "rule-based"
    return result
