
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SALES_FILE = PROCESSED_DIR / "sales_analysis.csv"
SEGMENTS_FILE = PROCESSED_DIR / "customer_segments.csv"
SEGMENT_SUMMARY_FILE = PROCESSED_DIR / "customer_segment_summary.csv"
PRIORITY_FILE = PROCESSED_DIR / "customer_priority_scores.csv"
CHURN_PREDICTIONS_FILE = (
    PROCESSED_DIR / "churn_model_predictions.csv"
)

app = FastAPI(
    title="Intelligent Sales & Customer Analytics API",
    description=(
        "API for sales summaries, customer segmentation, "
        "and rule-based customer prioritization."
    ),
    version="1.0.0",
)


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a processed CSV file and report useful errors."""

    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Required data file not found: {file_path.name}",
        )

    try:
        # Read the complete file when inferring column data types.
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


@app.get("/")
def root():
    """Return API welcome information."""
    return {
        "message": "Welcome to the Intelligent Sales & Customer Analytics API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """Check API status and availability of required data files."""
    files = {
    "sales_analysis": SALES_FILE,
    "customer_segments": SEGMENTS_FILE,
    "customer_segment_summary": SEGMENT_SUMMARY_FILE,
    "customer_priority_scores": PRIORITY_FILE,
    "churn_model_predictions": CHURN_PREDICTIONS_FILE,
}

    file_status = {
        name: path.exists()
        for name, path in files.items()
    }

    return {
        "status": "healthy" if all(file_status.values()) else "degraded",
        "data_files": file_status,
    }


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

    missing = required_sales_columns - set(sales.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Sales file is missing columns: {sorted(missing)}",
        )

    for column in ["CustomerID"]:
        if column not in segments.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Segments file is missing column: {column}",
            )

        if column not in priority.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Priority file is missing column: {column}",
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

    if "CustomerID" not in segments.columns:
        raise HTTPException(
            status_code=500,
            detail="CustomerID column is missing from the segments file.",
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

    missing = required_columns - set(priority.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Priority file is missing columns: {sorted(missing)}",
        )

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

    if "CustomerID" not in priority.columns:
        raise HTTPException(
            status_code=500,
            detail="CustomerID column is missing from the priority file.",
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

    missing = required_columns - set(predictions.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Churn predictions file is missing columns: {sorted(missing)}",
        )

    predictions[probability_columns[model]] = pd.to_numeric(
        predictions[probability_columns[model]],
        errors="coerce",
    )

    predictions = predictions.dropna(
        subset=[probability_columns[model]]
    )

    predictions = predictions.sort_values(
        probability_columns[model],
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

    if "CustomerID" not in predictions.columns:
        raise HTTPException(
            status_code=500,
            detail="CustomerID column is missing from churn predictions.",
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
