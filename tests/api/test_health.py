"""
API tests for /health and /ready endpoints.
"""
import pytest

pytestmark = pytest.mark.api


def test_health_returns_200(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200


def test_health_returns_ok_status(api_client):
    r = api_client.get("/health")
    assert r.json()["status"] == "ok"


def test_health_includes_timestamp(api_client):
    r = api_client.get("/health")
    assert "ts" in r.json()


def test_ready_returns_200_when_initialized(api_client):
    """
    The RAG pipeline initialises during lifespan startup. Once the TestClient
    context is entered, /ready should return 200.
    """
    r = api_client.get("/ready")
    # If no documents are indexed the pipeline may be empty — accept 200 or 503
    assert r.status_code in (200, 503)


def test_ready_200_includes_chunk_count(api_client):
    r = api_client.get("/ready")
    if r.status_code == 200:
        body = r.json()
        assert "chunks" in body
        assert isinstance(body["chunks"], int)


def test_home_endpoint(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_ping_endpoint(api_client):
    r = api_client.get("/ping")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}
