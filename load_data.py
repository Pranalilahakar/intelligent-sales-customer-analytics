
import pandas as pd
from getpass import getpass
from sqlalchemy import URL, create_engine, text

# 1. Read the cleaned dataset
input_file = "data/processed/online_retail_cleaned.csv"
df = pd.read_csv(input_file)

print("Original rows:", len(df))

# 2. Clean required fields
df["Description"] = df["Description"].replace(
    r"^\s*$", pd.NA, regex=True
)
df = df.dropna(subset=["Description"])

df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Description"] = df["Description"].astype(str).str.strip()
df["Country"] = df["Country"].astype(str).str.strip()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

# 3. Keep valid sales (exclude cancellations, returns, invalid prices)
df = df[
    (~df["InvoiceNo"].str.startswith("C"))
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
    & df["InvoiceDate"].notna()
].copy()

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print("Valid sales rows:", len(df))

# 4. Prepare customers (exclude missing customer IDs)
customer_rows = df[df["CustomerID"].notna()].copy()
customer_rows["CustomerID"] = customer_rows["CustomerID"].astype(int)

customers = (
    customer_rows[["CustomerID", "Country"]]
    .drop_duplicates(subset=["CustomerID"])
    .rename(columns={"CustomerID": "customer_id", "Country": "country"})
)

# 5. Prepare products
products = (
    df[["StockCode", "Description"]]
    .dropna()
    .drop_duplicates()
    .drop_duplicates(subset=["StockCode"])
    .rename(columns={"StockCode": "stock_code", "Description": "description"})
)

# 6. Prepare orders: one row per invoice
orders = (
    df.sort_values("InvoiceDate")
    .groupby("InvoiceNo", as_index=False)
    .agg(
        invoice_date=("InvoiceDate", "first"),
        customer_id=("CustomerID", "first"),
        country=("Country", "first"),
    )
    .rename(columns={"InvoiceNo": "invoice_no"})
)

orders["customer_id"] = orders["customer_id"].astype("Int64")

# 7. Prepare order items
order_items = df[
    ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "Revenue"]
].copy()

order_items = order_items.rename(
    columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Quantity": "quantity",
        "UnitPrice": "unit_price",
        "Revenue": "revenue",
    }
)

# 8. Connect to PostgreSQL securely
password = getpass("Enter your PostgreSQL password: ")

database_url = URL.create(
    "postgresql+psycopg2",
    username="postgres",
    password=password,
    host="localhost",
    port=5432,
    database="sales_analytics_db",
)

engine = create_engine(database_url)

try:
    with engine.begin() as connection:
        # Clear existing data in dependency order so reruns don't duplicate rows
        connection.execute(text("""
            TRUNCATE TABLE
                order_items, orders, products, customers
            RESTART IDENTITY CASCADE
        """))

        customers.to_sql(
            "customers", connection, if_exists="append",
            index=False, chunksize=5000
        )
        products.to_sql(
            "products", connection, if_exists="append",
            index=False, chunksize=5000
        )
        orders.to_sql(
            "orders", connection, if_exists="append",
            index=False, chunksize=5000
        )
        order_items.to_sql(
            "order_items", connection, if_exists="append",
            index=False, chunksize=5000
        )

    print("\nData imported successfully!")
    print("Customers:", len(customers))
    print("Products:", len(products))
    print("Orders:", len(orders))
    print("Order items:", len(order_items))

except Exception as error:
    print("\nImport failed:", error)
    print("The database transaction will be rolled back.")

finally:
    engine.dispose()
