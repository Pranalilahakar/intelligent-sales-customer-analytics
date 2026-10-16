
import pandas as pd
from pathlib import Path

# File paths
input_file = Path("data/raw/Online Retail.xlsx")
output_file = Path("data/processed/online_retail_cleaned.csv")

# Load the original dataset
df = pd.read_excel(input_file)

print("Original shape:", df.shape)

# 1. Remove exact duplicate rows
df = df.drop_duplicates()

# 2. Remove rows without a product description
df = df.dropna(subset=["Description"])

# 3. Standardize text fields
df["Description"] = df["Description"].str.strip()
df["Country"] = df["Country"].str.strip()

# 4. Convert column data types
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype("Int64")

# 5. Create a revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# 6. Create the output directory if needed
output_file.parent.mkdir(parents=True, exist_ok=True)

# 7. Save the cleaned dataset
df.to_csv(output_file, index=False)

# 8. Display cleaning results
print("Cleaned shape:", df.shape)
print("Missing values after cleaning:")
print(df.isnull().sum())
print("Saved cleaned data to:", output_file)
