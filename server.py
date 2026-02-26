# server.py
"""
Entry point for running locally or in Docker.

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from app.main import app

# Optional: allow `python server.py` for local dev convenience
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)