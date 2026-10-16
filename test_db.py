
import os
from getpass import getpass

import pytest
from sqlalchemy import URL, create_engine, text


def test_postgresql_connection():
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        if not os.getenv("RUN_DB_TEST"):
            pytest.skip(
                "Database test skipped. Set RUN_DB_TEST=1 to run it."
            )

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

        assert database == "sales_analytics_db"
        print("Connected database:", database)

    finally:
        engine.dispose()
