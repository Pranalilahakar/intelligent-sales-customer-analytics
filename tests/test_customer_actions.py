
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_customer_actions():
    response = client.get("/customer-actions?limit=3")

    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert data["count"] <= 3
    assert len(data["results"]) == data["count"]

    if data["results"]:
        assert "CustomerID" in data["results"][0]
        assert "Recommended_Action" in data["results"][0]
        assert "Action_Priority" in data["results"][0]


def test_filter_high_priority_actions():
    response = client.get("/customer-actions?action_priority=High&limit=100")

    assert response.status_code == 200

    data = response.json()
    assert all(
        row["Action_Priority"] == "High"
        for row in data["results"]
    )


def test_invalid_action_priority():
    response = client.get("/customer-actions?action_priority=Urgent")

    assert response.status_code == 400


def test_filter_by_recommended_action():
    action = "High-Value Win-Back Campaign"

    response = client.get(
        "/customer-actions",
        params={"recommended_action": action, "limit": 100},
    )

    assert response.status_code == 200

    data = response.json()
    assert all(
        row["Recommended_Action"] == action
        for row in data["results"]
    )


def test_invalid_recommended_action():
    response = client.get(
        "/customer-actions",
        params={"recommended_action": "Unknown Campaign"},
    )

    assert response.status_code == 400


def test_get_single_customer_action():
    response = client.get("/customer-actions?limit=1")

    assert response.status_code == 200
    results = response.json()["results"]

    if not results:
        return

    customer_id = results[0]["CustomerID"]
    detail_response = client.get(f"/customer-actions/{customer_id}")

    assert detail_response.status_code == 200

    data = detail_response.json()
    assert "Recommended_Action" in data
    assert data["recommendation_type"] == "rule-based"


def test_unknown_customer_action():
    response = client.get("/customer-actions/999999999")

    assert response.status_code == 404
