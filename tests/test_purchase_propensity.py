
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_list_purchase_propensity_predictions():
    response = client.get(
        "/purchase-propensity",
        params={"limit": 3},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["limit"] == 3
    assert data["count"] <= 3
    assert isinstance(data["results"], list)

    for customer in data["results"]:
        assert "CustomerID" in customer
        assert "Purchase_Propensity_Score" in customer
        assert 0 <= customer["Purchase_Propensity_Score"] <= 100


def test_purchase_propensity_live_scoring():
    payload = {
        "Recency_Days": 59.63,
        "Total_Invoices": 5,
        "Total_Revenue": 2790.86,
        "Total_Transactions": 124,
        "Average_Quantity": 12.82,
        "Average_Order_Value": 558.17,
    }

    response = client.post(
        "/purchase-propensity/score",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_purchased_future_window"] in (0, 1)
    assert 0 <= data["purchase_propensity_score"] <= 100
    assert data["score_scale"] == "0-100"
    assert data["selected_model"] == "Logistic Regression"
    assert data["target"] == "Purchased_Future_Window"
    assert "note" in data


def test_purchase_propensity_rejects_missing_features():
    response = client.post(
        "/purchase-propensity/score",
        json={},
    )

    assert response.status_code == 422


def test_purchase_propensity_rejects_negative_recency():
    payload = {
        "Recency_Days": -1,
        "Total_Invoices": 5,
        "Total_Revenue": 2790.86,
        "Total_Transactions": 124,
        "Average_Quantity": 12.82,
        "Average_Order_Value": 558.17,
    }

    response = client.post(
        "/purchase-propensity/score",
        json=payload,
    )

    assert response.status_code == 422
