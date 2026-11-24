"""GitHub sign-in (OAuth) and role-based access control.

Sign-in flow
    1. /auth/login      makes a random `state`, keeps it in the session cookie and
                        sends the browser to GitHub's consent page
    2. GitHub           the user approves; GitHub redirects back with a one-time `code`
    3. /auth/callback   checks `state` (blocks forged logins), swaps the `code` for a
                        token, reads the user's public profile, then FORGETS the token
    4.                  the session cookie (signed, HttpOnly) stores login + expiry only
    5. /auth/logout     clears the session

Roles (lowest to highest): viewer < editor < admin
    viewer  ask questions, search, see the file list
    editor  + upload and replace documents (next feature)
    admin   + delete documents, re-ingest, read logs and reports

A user's role comes from the GitHub username lists in .env (RAG_ADMINS,
RAG_EDITORS, RAG_VIEWERS) and is looked up on EVERY request, so removing someone
from a list takes effect on the next server start without waiting for their
session to expire. Users on no list get RAG_DEFAULT_ROLE ("none" = refused).
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from ingestion.settings import Settings

log = logging.getLogger("api.auth")

ROLES = ("viewer", "editor", "admin")
GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"


class AuthConfig:
    """Login settings, read once at start-up. Tests replace `config` with their own."""

    def __init__(self, settings: Settings):
        self.mode = settings.auth_mode
        self.client_id = settings.github_client_id.strip()
        self.client_secret = settings.github_client_secret.strip()
        self.session_secret = settings.session_secret.strip()
        self.public_url = settings.public_url.rstrip("/")
        self.session_seconds = settings.session_hours * 3600
        self.default_role = settings.default_role
        self.audit_file = settings.reports_dir / "access.jsonl"
        self.role_of: dict[str, str] = {}
        for role, names in (("viewer", settings.viewers), ("editor", settings.editors),
                            ("admin", settings.admins)):  # admin last, so it wins
            for name in names.split(","):
                if name.strip():
                    self.role_of[name.strip().lower()] = role

    @property
    def callback_url(self) -> str:
        return f"{self.public_url}/auth/callback"

    def missing(self) -> list[str]:
        """Settings GitHub login can't work without."""
        if self.mode != "github":
            return []
        needed = {"RAG_GITHUB_CLIENT_ID": self.client_id,
                  "RAG_GITHUB_CLIENT_SECRET": self.client_secret,
                  "RAG_SESSION_SECRET": self.session_secret}
        problems = [name for name, value in needed.items() if not value]
        if self.session_secret and len(self.session_secret) < 32:
            problems.append("RAG_SESSION_SECRET (must be at least 32 characters)")
        return problems

    def role_for(self, login: str) -> str | None:
        role = self.role_of.get(login.lower())
        if role:
            return role
        return "viewer" if self.default_role == "viewer" else None


config = AuthConfig(Settings())
router = APIRouter()


# ===================================================================== audit
def audit(event: str, login: str | None, **details) -> None:
    """Append who did what to data/reports/access.jsonl."""
    try:
        config.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with config.audit_file.open("a") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "event": event, "user": login, **details}) + "\n")
    except OSError as e:
        log.warning("could not write access.jsonl: %s", e)


# =========================================================== who is calling?
LOCAL_USER = {"login": "local", "name": "Local user (sign-in off)", "avatar_url": "", "role": "admin"}


def current_user(request: Request) -> dict:
    """The signed-in user with their CURRENT role, or HTTP 401/403/503."""
    if config.mode == "off":
        return LOCAL_USER
    if config.missing():
        raise HTTPException(503, detail="GitHub sign-in is not configured. Missing in .env: "
                                        + ", ".join(config.missing()))
    user = request.session.get("user")
    if not user or user.get("expires", 0) < time.time():
        request.session.pop("user", None)
        raise HTTPException(401, detail="Please sign in with GitHub.")
    role = config.role_for(user["login"])
    if role is None:  # removed from the lists since signing in
        request.session.pop("user", None)
        raise HTTPException(403, detail=f"@{user['login']} no longer has access.")
    return {**user, "role": role}


def require(role: str):
    """FastAPI dependency: the caller must have at least this role."""
    needed = ROLES.index(role)

    def check(user: dict = Depends(current_user)) -> dict:
        if ROLES.index(user["role"]) < needed:
            raise HTTPException(403, detail=f"This needs the '{role}' role; you are '{user['role']}'.")
        return user

    return check


# ================================================================== routes
@router.get("/auth/login")
def login(request: Request):
    if config.mode == "off":
        return RedirectResponse("/")
    if config.missing():
        raise HTTPException(503, detail="GitHub sign-in is not configured. Missing in .env: "
                                        + ", ".join(config.missing()))
    # GitHub sends the user back to PUBLIC_URL. The session cookie belongs to the
    # address the page was opened on, so "localhost" vs "127.0.0.1" would lose it
    # and the sign-in would fail. Start the sign-in on the PUBLIC_URL address.
    public = urlparse(config.public_url)
    if request.url.netloc != public.netloc:
        return RedirectResponse(f"{config.public_url}/auth/login")
    state = secrets.token_urlsafe(24)
    request.session["oauth_state"] = state
    query = urlencode({"client_id": config.client_id, "redirect_uri": config.callback_url,
                       "scope": "read:user", "state": state, "allow_signup": "false"})
    return RedirectResponse(f"{GITHUB_AUTHORIZE}?{query}")


@router.get("/auth/callback")
def callback(request: Request, code: str = "", state: str = "", error: str = ""):
    expected = request.session.pop("oauth_state", None)
    if error:  # e.g. the user pressed "Cancel" on GitHub
        return _back_to_app("cancelled")
    if not expected or not state or not secrets.compare_digest(expected, state):
        audit("login_rejected", None, reason="state mismatch")
        return _back_to_app("expired")  # stale tab, replayed or forged callback
    if not code:
        return _back_to_app("failed")

    try:
        token = exchange_code(code)
        profile = fetch_user(token)
    except GitHubError as e:
        log.warning("GitHub sign-in failed: %s", e)
        audit("login_failed", None, reason=str(e))
        return _back_to_app("failed")

    login_name = profile["login"]
    role = config.role_for(login_name)
    if role is None:
        audit("login_refused", login_name, reason="not on any role list")
        return _back_to_app("not_allowed", login_name)

    request.session["user"] = {
        "login": login_name,
        "name": profile.get("name") or login_name,
        "avatar_url": profile.get("avatar_url", ""),
        "expires": int(time.time()) + config.session_seconds,
    }
    audit("login", login_name, role=role)
    return RedirectResponse("/", status_code=303)


@router.post("/auth/logout", status_code=204)
def logout(request: Request):
    user = request.session.get("user")
    request.session.clear()
    if user:
        audit("logout", user["login"])
    return Response(status_code=204)


@router.get("/api/me")
def me(user: dict = Depends(current_user)):
    return {"login": user["login"], "name": user["name"], "avatar_url": user["avatar_url"],
            "role": user["role"], "auth": config.mode}


def _back_to_app(reason: str, login_name: str = "") -> RedirectResponse:
    query = urlencode({"login_error": reason, **({"user": login_name} if login_name else {})})
    return RedirectResponse(f"/?{query}", status_code=303)


# ============================================================ GitHub calls
class GitHubError(Exception):
    pass


def exchange_code(code: str) -> str:
    """Swap the one-time code for an access token (used once, never stored)."""
    try:
        r = httpx.post(GITHUB_TOKEN, headers={"Accept": "application/json"}, timeout=15, data={
            "client_id": config.client_id, "client_secret": config.client_secret,
            "code": code, "redirect_uri": config.callback_url})
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise GitHubError(f"token request failed: {e}") from e
    if "access_token" not in data:
        raise GitHubError(f"GitHub refused the code: {data.get('error_description') or data.get('error')}")
    return data["access_token"]


def fetch_user(token: str) -> dict:
    try:
        r = httpx.get(GITHUB_USER, timeout=15, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        r.raise_for_status()
        profile = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise GitHubError(f"could not read the GitHub profile: {e}") from e
    if not profile.get("login"):
        raise GitHubError("GitHub profile has no login")
    return profile
