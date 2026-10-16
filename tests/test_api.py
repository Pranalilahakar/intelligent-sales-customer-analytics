
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_sales_summary():
    response = client.get("/summary")
    assert response.status_code == 200

    data = response.json()
    assert data["total_revenue"] > 0
    assert data["total_quantity"] > 0
    assert data["unique_invoices"] > 0
    assert data["customers_analyzed"] == 4338
    assert data["customers_scored"] == 4338


def test_list_segments():
    response = client.get("/segments")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 4
    assert len(data["segments"]) == 4


def test_customer_segment_found():
    response = client.get("/segments/14911")
    assert response.status_code == 200
    assert str(response.json()["CustomerID"]).replace(".0", "") == "14911"


def test_customer_segment_not_found():
    response = client.get("/segments/999999999")
    assert response.status_code == 404


def test_priority_list():
    response = client.get("/priority")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 20
    assert len(data["results"]) == 20
    assert data["results"][0]["Priority_Score"] >= data["results"][-1]["Priority_Score"]


def test_priority_category_filter():
    response = client.get(
        "/priority",
        params={"category": "High Priority", "limit": 10},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["count"] <= 10
    assert all(
        item["Priority_Category"] == "High Priority"
        for item in data["results"]
    )


def test_invalid_priority_category():
    response = client.get(
        "/priority",
        params={"category": "Invalid Category"},
    )
    assert response.status_code == 400


def test_customer_priority_found():
    response = client.get("/priority/14911")
    assert response.status_code == 200

    data = response.json()
    assert data["Priority_Category"] == "High Priority"
    assert data["Priority_Score"] > 0


def test_customer_priority_not_found():
    response = client.get("/priority/999999999")
    assert response.status_code == 404
