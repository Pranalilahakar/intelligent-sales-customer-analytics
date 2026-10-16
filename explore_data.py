
import pandas as pd
from pathlib import Path

# 1. Locate the Excel dataset
file_path = Path("data/raw/Online Retail.xlsx")

# 2. Check whether the file exists
if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {file_path.resolve()}"
    )

# 3. Load the dataset into a DataFrame
df = pd.read_excel(file_path)

# 4. Display the first five records
print("\n--- First 5 Rows ---")
print(df.head())

# 5. Check the number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# 6. Display column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 7. Check data types
print("\n--- Data Information ---")
df.info()

# 8. Count missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 9. Count duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())
