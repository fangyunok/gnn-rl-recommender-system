from fastapi.testclient import TestClient

from gnn_rl_recommender.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommendations_are_ranked_and_bounded() -> None:
    response = client.get("/v1/recommendations/3?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 5
    assert len({item["item_id"] for item in payload["items"]}) == 5
    scores = [item["score"] for item in payload["items"]]
    assert scores == sorted(scores, reverse=True)


def test_unknown_user() -> None:
    response = client.get("/v1/recommendations/999")
    assert response.status_code == 404

