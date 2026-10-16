
import pandas as pd
from pathlib import Path

# Load the cleaned dataset
file_path = Path("data/processed/online_retail_cleaned.csv")
df = pd.read_csv(file_path)

# Convert columns to correct data types
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype("Int64")

# Treat whitespace-only descriptions as missing
df["Description"] = df["Description"].replace(
    r"^\s*$", pd.NA, regex=True
)

# Remove rows with missing descriptions
df = df.dropna(subset=["Description"])

print("\n--- DATA QUALITY REPORT ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Missing descriptions:", df["Description"].isna().sum())
print("Missing customer IDs:", df["CustomerID"].isna().sum())
print("Missing values by column:")
print(df.isnull().sum())

# Separate cancelled invoices for inspection
cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
print("\n--- TRANSACTION CHECKS ---")
print("Cancellation invoice rows:", cancelled.sum())
print("Negative quantity rows:", (df["Quantity"] < 0).sum())
print("Zero or negative price rows:", (df["UnitPrice"] <= 0).sum())

# Create a sales-analysis dataset.
# Exclude cancellations, returns, and non-positive prices
sales = df[
    (~cancelled)
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
].copy()

sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]

print("\n--- SALES SUMMARY ---")
print("Valid sales rows:", len(sales))
print("Total revenue: £", round(sales["Revenue"].sum(), 2))
print("Unique invoices:", sales["InvoiceNo"].nunique())
print("Unique identified customers:", sales["CustomerID"].nunique())

# Monthly revenue
sales["Month"] = sales["InvoiceDate"].dt.to_period("M")

monthly_sales = (
    sales.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

print("\n--- MONTHLY REVENUE ---")
print(monthly_sales.to_string())

# Top 10 products by revenue
top_products = (
    sales.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n--- TOP 10 PRODUCTS BY REVENUE ---")
print(top_products.to_string())

# Save analysis-ready sales data
output_file = Path("data/processed/sales_analysis.csv")
sales.to_csv(output_file, index=False)

print("\nAnalysis-ready data saved to:", output_file)
