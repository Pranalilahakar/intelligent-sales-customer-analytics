
from getpass import getpass

from sqlalchemy import URL, create_engine, text

password = getpass("Enter your PostgreSQL password: ")

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password=password,
    host="localhost",
    port=5432,
    database="sales_analytics_db",
)

engine = create_engine(database_url)

try:
    with engine.connect() as connection:
        database = connection.execute(
            text("SELECT current_database();")
        ).scalar_one()

        print("Connection successful!")
        print("Connected database:", database)

except Exception as error:
    print("Connection failed:", error)

finally:
    engine.dispose()
