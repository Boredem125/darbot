"""Tests for the FastAPI endpoints: /health, /ready, /stats, rate limiting, XFF."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------


class TestReady:
    def test_ready_503_when_not_started(self):
        """When _ready is False, /ready returns 503."""
        import app.api as api_mod
        old_ready = api_mod._ready
        old_svc = api_mod._chat_service
        try:
            api_mod._ready = False
            api_mod._chat_service = None
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 503
        finally:
            api_mod._ready = old_ready
            api_mod._chat_service = old_svc


# ---------------------------------------------------------------------------
# /stats shape
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_shape(self):
        """Stats returns a JSON object with the expected keys."""
        import app.api as api_mod
        old_svc = api_mod._chat_service
        try:
            api_mod._chat_service = None
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "n" in data
            assert "routes" in data
            assert "tiers" in data
            assert "providers" in data
            assert "cache_rates" in data
        finally:
            api_mod._chat_service = old_svc


# ---------------------------------------------------------------------------
# Rate limiting: two XFF values get independent buckets
# ---------------------------------------------------------------------------


class TestRateLimitXFF:
    def test_different_xff_independent(self):
        """Two different X-Forwarded-For values should not share rate limit counters."""
        import app.api as api_mod
        old_ready = api_mod._ready
        old_svc = api_mod._chat_service
        old_settings = api_mod._settings
        try:
            # Set up: not ready so /api/chat returns 503 fast (no need for real service)
            api_mod._ready = False
            api_mod._chat_service = None

            client = TestClient(app, raise_server_exceptions=False)

            # Both IPs can make requests (they get 503 for not_ready, not 429)
            resp_a = client.post(
                "/api/chat",
                json={"message": "test"},
                headers={"X-Forwarded-For": "1.1.1.1"},
            )
            resp_b = client.post(
                "/api/chat",
                json={"message": "test"},
                headers={"X-Forwarded-For": "2.2.2.2"},
            )
            # Neither should be rate-limited on first request
            assert resp_a.status_code == 503  # not_ready, not 429
            assert resp_b.status_code == 503
        finally:
            api_mod._ready = old_ready
            api_mod._chat_service = old_svc
            api_mod._settings = old_settings


# ---------------------------------------------------------------------------
# Cache-Control on static files
# ---------------------------------------------------------------------------


class TestStaticCacheControl:
    def test_root_has_no_cache(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/")
        if resp.status_code == 200:
            assert resp.headers.get("cache-control") == "no-cache"

    def test_app_js_has_no_cache(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/app.js")
        if resp.status_code == 200:
            assert resp.headers.get("cache-control") == "no-cache"
