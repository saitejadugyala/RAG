"""GitHub sign-in and role checks. GitHub itself is replaced by fakes - no network,
no real accounts."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.auth as auth
import api.main as api_main
from ingestion import IngestionPipeline, Settings
from retrieval import RAGPipeline

SECRET = "x" * 40
PROFILES = {
    "code-alice": {"login": "alice", "name": "Alice A", "avatar_url": "https://example.test/a.png"},
    "code-bob": {"login": "Bob", "name": "Bob B", "avatar_url": ""},
    "code-mallory": {"login": "mallory", "name": "Mallory", "avatar_url": ""},
}


@pytest.fixture(scope="module")
def rag(tmp_path_factory, rag_docs):
    folder = tmp_path_factory.mktemp("auth")
    IngestionPipeline(Settings(data_dir=folder), fake_embeddings=True).run([rag_docs / "rag_retrieval_notes.txt"])
    return RAGPipeline(Settings(data_dir=folder), fake=True)


@pytest.fixture
def github_config(tmp_path, monkeypatch):
    config = auth.AuthConfig(Settings(
        data_dir=tmp_path, auth_mode="github", github_client_id="client-123",
        github_client_secret="shh", session_secret=SECRET, admins="alice", viewers="bob",
        public_url="http://testserver"))
    monkeypatch.setattr(auth, "config", config)
    # Fake GitHub: the code says who signs in; the token is a recognisable string
    monkeypatch.setattr(auth, "exchange_code", lambda code: f"gho_SECRET_TOKEN_FOR_{code}")
    monkeypatch.setattr(auth, "fetch_user", lambda token: PROFILES[token.split("_FOR_")[1]])
    return config


@pytest.fixture
def client(github_config, rag, monkeypatch):
    state = api_main.State()
    state.rag = rag
    monkeypatch.setattr(api_main, "state", state)
    with TestClient(api_main.app) as c:
        yield c


def sign_in(client, code: str):
    """Walk through the real redirect flow and return the final redirect."""
    to_github = client.get("/auth/login", follow_redirects=False)
    assert to_github.status_code in (302, 307)
    url = urlparse(to_github.headers["location"])
    query = parse_qs(url.query)
    assert url.netloc == "github.com" and query["client_id"] == ["client-123"]
    assert query["scope"] == ["read:user"]
    return client.get(f"/auth/callback?code={code}&state={query['state'][0]}", follow_redirects=False)


# ------------------------------------------------------------------ signing in
def test_admin_signs_in_and_can_ask(client):
    back = sign_in(client, "code-alice")
    assert back.status_code == 303 and back.headers["location"] == "/"
    me = client.get("/api/me").json()
    assert me["login"] == "alice" and me["role"] == "admin"
    r = client.post("/api/ask", json={"question": "Reciprocal Rank Fusion k value"})
    assert r.status_code == 200


def test_usernames_are_case_insensitive(client):
    sign_in(client, "code-bob")  # GitHub says "Bob", the list says "bob"
    assert client.get("/api/me").json()["role"] == "viewer"


def test_user_not_on_any_list_is_refused(client):
    back = sign_in(client, "code-mallory")
    assert "login_error=not_allowed" in back.headers["location"]
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/ask", json={"question": "anything at all"}).status_code == 401


def test_default_role_viewer_lets_any_github_user_in(client, github_config):
    github_config.default_role = "viewer"
    sign_in(client, "code-mallory")
    assert client.get("/api/me").json()["role"] == "viewer"


def test_forged_or_stale_state_is_rejected(client):
    client.get("/auth/login", follow_redirects=False)
    back = client.get("/auth/callback?code=code-alice&state=forged", follow_redirects=False)
    assert "login_error=expired" in back.headers["location"]
    assert client.get("/api/me").status_code == 401


def test_callback_without_login_attempt_is_rejected(client):
    back = client.get("/auth/callback?code=code-alice&state=anything", follow_redirects=False)
    assert "login_error=expired" in back.headers["location"]


def test_cancel_on_github(client):
    client.get("/auth/login", follow_redirects=False)
    back = client.get("/auth/callback?error=access_denied", follow_redirects=False)
    assert "login_error=cancelled" in back.headers["location"]


def test_github_failure_is_handled(client, monkeypatch):
    def broken(code):
        raise auth.GitHubError("GitHub is down")

    monkeypatch.setattr(auth, "exchange_code", broken)
    assert "login_error=failed" in sign_in(client, "code-alice").headers["location"]


# -------------------------------------------------------------- the session
def test_github_token_is_never_stored_in_the_cookie(client):
    sign_in(client, "code-alice")
    cookie = client.cookies.get("rag_session")
    payload = base64.b64decode(cookie.split(".")[0] + "==").decode()
    assert "alice" in payload and "SECRET_TOKEN" not in payload and "gho_" not in payload


def test_session_expires(client, monkeypatch):
    sign_in(client, "code-alice")
    future = auth.time.time() + 9 * 3600  # sessions last 8 hours
    monkeypatch.setattr(auth.time, "time", lambda: future)
    assert client.get("/api/me").status_code == 401


def test_removed_user_loses_access_immediately(client, github_config):
    sign_in(client, "code-bob")
    assert client.get("/api/me").status_code == 200
    github_config.role_of.pop("bob")  # admin removed bob from RAG_VIEWERS (and restarted)
    assert client.get("/api/me").status_code == 403


def test_sign_out(client):
    sign_in(client, "code-alice")
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/api/me").status_code == 401


# ------------------------------------------------------------- roles + config
def test_roles_are_ordered():
    check_editor = auth.require("editor")
    with pytest.raises(HTTPException) as refused:
        check_editor(user={"role": "viewer"})
    assert refused.value.status_code == 403
    assert check_editor(user={"role": "editor"})["role"] == "editor"
    assert check_editor(user={"role": "admin"})["role"] == "admin"


def test_admin_list_wins_over_viewer_list(tmp_path):
    config = auth.AuthConfig(Settings(data_dir=tmp_path, viewers="carol", admins="carol"))
    assert config.role_for("carol") == "admin"


def test_missing_configuration_returns_503_with_the_names(tmp_path, monkeypatch, rag):
    monkeypatch.setattr(auth, "config", auth.AuthConfig(Settings(
        data_dir=tmp_path, auth_mode="github", github_client_id="", github_client_secret="",
        session_secret="short")))  # every value set here, so the real .env can't leak in
    with TestClient(api_main.app) as c:
        r = c.get("/api/me")
        assert r.status_code == 503
        for name in ("RAG_GITHUB_CLIENT_ID", "RAG_GITHUB_CLIENT_SECRET", "at least 32 characters"):
            assert name in r.json()["detail"]


def test_web_page_is_public_but_api_is_not(client):
    assert client.get("/").status_code == 200
    for path in ("/api/health", "/api/files"):
        assert client.get(path).status_code == 401


def test_logins_and_questions_are_audited(client, github_config):
    sign_in(client, "code-alice")
    client.post("/api/ask", json={"question": "Reciprocal Rank Fusion k value"})
    sign_in(client, "code-mallory")
    events = [json.loads(line) for line in github_config.audit_file.read_text().splitlines()]
    assert {"event": "login", "user": "alice"}.items() <= events[0].items()
    assert any(e["event"] == "ask" and e["user"] == "alice" for e in events)
    assert any(e["event"] == "login_refused" and e["user"] == "mallory" for e in events)


def test_login_moves_to_the_public_address_first(client, github_config):
    github_config.public_url = "http://127.0.0.1:8000"  # page opened on another address
    r = client.get("/auth/login", follow_redirects=False)
    assert r.headers["location"] == "http://127.0.0.1:8000/auth/login"
    assert "oauth_state" not in client.cookies.get("rag_session", "")
