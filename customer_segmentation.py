
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = Path("data/processed/sales_analysis.csv")
output_dir = Path("data/processed")

customer_segments_file = output_dir / "customer_segments.csv"
segment_summary_file = output_dir / "customer_segment_summary.csv"
model_file = output_dir / "customer_segmentation_model.joblib"
scaler_file = output_dir / "customer_segmentation_scaler.joblib"

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD AND VALIDATE SALES DATA
# ============================================================

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file}\n"
        "Run your existing analyze_data.py script first."
    )

sales = pd.read_csv(input_file, low_memory=False)

required_columns = [
    "CustomerID",
    "InvoiceNo",
    "InvoiceDate",
    "Revenue"
]

missing_columns = [
    column for column in required_columns
    if column not in sales.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

sales["InvoiceDate"] = pd.to_datetime(
    sales["InvoiceDate"], errors="coerce"
)

sales["CustomerID"] = pd.to_numeric(
    sales["CustomerID"], errors="coerce"
)

sales["Revenue"] = pd.to_numeric(
    sales["Revenue"], errors="coerce"
)

# Remove rows with missing or invalid required values
sales = sales.dropna(
    subset=required_columns
).copy()

# Exclude records without valid positive customer IDs,
# positive revenue, or invoice dates
sales = sales[
    (sales["CustomerID"] > 0)
    & (sales["Revenue"] > 0)
].copy()

sales["CustomerID"] = sales["CustomerID"].astype("int64")
sales["InvoiceNo"] = sales["InvoiceNo"].astype(str)

if sales.empty:
    raise ValueError(
        "No valid sales records remain after data validation."
    )

print("\n--- SALES DATA CHECK ---")
print("Valid sales rows:", len(sales))
print("Unique customers:", sales["CustomerID"].nunique())
print("Unique invoices:", sales["InvoiceNo"].nunique())

# ============================================================
# 3. CALCULATE RFM FEATURES
# ============================================================

# Use one day after the latest transaction as the reference date
snapshot_date = (
    sales["InvoiceDate"].max() + pd.Timedelta(days=1)
)

rfm = sales.groupby("CustomerID").agg(
    Recency=(
        "InvoiceDate",
        lambda dates: (snapshot_date - dates.max()).days
    ),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

# Keep customers with valid RFM values
rfm = rfm.replace([np.inf, -np.inf], np.nan)
rfm = rfm.dropna(
    subset=["Recency", "Frequency", "Monetary"]
)

rfm = rfm[
    (rfm["Recency"] >= 0)
    & (rfm["Frequency"] > 0)
    & (rfm["Monetary"] > 0)
].copy()

if len(rfm) < 4:
    raise ValueError(
        "At least four valid customers are required "
        "for four-cluster K-Means."
    )

print("\n--- RFM SUMMARY ---")
print("Customers analyzed:", len(rfm))
print("Average recency:", round(rfm["Recency"].mean(), 2))
print("Average frequency:", round(rfm["Frequency"].mean(), 2))
print("Average monetary value: £", round(rfm["Monetary"].mean(), 2))

# ============================================================
# 4. PREPARE FEATURES AND TRAIN K-MEANS
# ============================================================

features = ["Recency", "Frequency", "Monetary"]

# Log transformation reduces the impact of highly skewed values
X = np.log1p(rfm[features])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# ============================================================
# 5. CREATE CLUSTER SUMMARY
# ============================================================

segment_summary = rfm.groupby("Cluster").agg(
    Total_Customers=("CustomerID", "count"),
    Average_Recency_Days=("Recency", "mean"),
    Average_Invoices=("Frequency", "mean"),
    Average_Revenue=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum")
).reset_index()

numeric_columns = [
    "Average_Recency_Days",
    "Average_Invoices",
    "Average_Revenue",
    "Total_Revenue"
]

segment_summary[numeric_columns] = (
    segment_summary[numeric_columns].round(2)
)

# ============================================================
# 6. ASSIGN BUSINESS-FRIENDLY SEGMENT NAMES
# ============================================================

# Determine cluster characteristics from the actual averages.
# These labels are heuristic interpretations, not predictions
# of future customer behavior.

segment_names = {}

for _, row in segment_summary.iterrows():
    cluster = int(row["Cluster"])
    recency = row["Average_Recency_Days"]
    frequency = row["Average_Invoices"]
    monetary = row["Average_Revenue"]

    if (
        frequency >= segment_summary["Average_Invoices"].median()
        and monetary >= segment_summary["Average_Revenue"].median()
        and recency <= segment_summary["Average_Recency_Days"].median()
    ):
        name = "High-Value Active"
    elif (
        recency > segment_summary["Average_Recency_Days"].median()
        and frequency <= segment_summary["Average_Invoices"].median()
    ):
        name = "Inactive / Low-Frequency"
    elif (
        recency <= segment_summary["Average_Recency_Days"].median()
        and monetary < segment_summary["Average_Revenue"].median()
    ):
        name = "Recent Low-Spend"
    else:
        name = "Regular Customers"

    segment_names[cluster] = name

segment_summary["Segment_Name"] = (
    segment_summary["Cluster"].map(segment_names)
)

# Add the business-friendly name to every customer's record
rfm["Segment_Name"] = rfm["Cluster"].map(segment_names)

# ============================================================
# 7. SAVE CUSTOMER-LEVEL AND SUMMARY OUTPUTS
# ============================================================

rfm.to_csv(customer_segments_file, index=False)
segment_summary.to_csv(segment_summary_file, index=False)

# Save the trained model and fitted scaler
joblib.dump(kmeans, model_file)
joblib.dump(scaler, scaler_file)

# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n--- CUSTOMER SEGMENT SUMMARY ---")
print(segment_summary.to_string(index=False))

print("\n--- CUSTOMER DISTRIBUTION ---")
print(
    rfm.groupby("Segment_Name")["CustomerID"]
    .count()
    .sort_values(ascending=False)
    .to_string()
)

print("\n--- FILES SAVED SUCCESSFULLY ---")
print("Customer-level results:", customer_segments_file)
print("Power BI summary:", segment_summary_file)
print("Trained K-Means model:", model_file)
print("Fitted feature scaler:", scaler_file)

print("\nCustomer segmentation pipeline completed.")
