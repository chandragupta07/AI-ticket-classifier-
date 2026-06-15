import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """Fixture to provide a TestClient with the application's lifespan events executed."""
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    """Test that the root endpoint serves the HTML dashboard UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SupportMind AI" in response.text

def test_health_check(client):
    """Test that the health endpoint returns correct status and message."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "AI Support Ticket Classifier API is running" in data["message"]

def test_predict_billing(client):
    """Test classification, priority, and sentiment of a billing support ticket."""
    payload = {"text": "I was charged twice for my subscription this month. Can I get a refund?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "billing"
    assert isinstance(data["confidence"], float)
    assert data["confidence"] > 0.0
    assert data["priority"] == "high"
    assert isinstance(data["priority_confidence"], float)
    assert data["priority_confidence"] > 0.0
    assert isinstance(data["sentiment"], str)
    assert isinstance(data["sentiment_confidence"], float)
    assert isinstance(data["reasoning"], list)
    assert len(data["reasoning"]) > 0
    assert isinstance(data["suggested_action"], str)
    assert len(data["suggested_action"]) > 0

def test_predict_password_reset(client):
    """Test classification, priority, and sentiment of a password reset ticket."""
    payload = {"text": "How do I reset my password? I forgot it."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "password_reset"
    assert isinstance(data["confidence"], float)
    assert data["confidence"] > 0.0
    assert isinstance(data["priority"], str)
    assert isinstance(data["priority_confidence"], float)
    assert data["priority_confidence"] > 0.0
    assert isinstance(data["sentiment"], str)
    assert isinstance(data["sentiment_confidence"], float)
    assert isinstance(data["reasoning"], list)
    assert len(data["reasoning"]) > 0
    assert isinstance(data["suggested_action"], str)
    assert len(data["suggested_action"]) > 0

def test_predict_empty_text(client):
    """Test that empty or whitespace-only text returns a 400 Bad Request."""
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Input ticket text cannot be empty or whitespace only."

def test_predict_invalid_schema(client):
    """Test that missing required fields in payload returns 422 Unprocessable Entity."""
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
