"""Start the web app: FastAPI backend + HTML/CSS/JS frontend.

    python serve.py                 # http://127.0.0.1:8000
    python serve.py --port 8080
    python serve.py --reload        # restart automatically when code changes

Only listens on this computer (127.0.0.1). Open the app at the address it prints
(127.0.0.1, not localhost - GitHub sign-in returns to that exact address).
"""
import argparse
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import uvicorn  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # GitHub sends users back to RAG_PUBLIC_URL; default it to this port.
    # (.env is read first, so a RAG_PUBLIC_URL set there still wins.)
    load_dotenv()
    os.environ.setdefault("RAG_PUBLIC_URL", f"http://127.0.0.1:{args.port}")
    url = os.environ["RAG_PUBLIC_URL"].rstrip("/")
    print(f"RAG web app: {url}   (API docs: {url}/docs, stop: Ctrl+C)")
    if os.getenv("RAG_AUTH", "github") == "github":
        print(f"GitHub OAuth app callback URL must be: {url}/auth/callback")
    uvicorn.run("api.main:app", host="127.0.0.1", port=args.port, reload=args.reload,
                log_level="info")


if __name__ == "__main__":
    main()
