# app/services/firebase_legacy.py
from __future__ import annotations

import os
import io
import json
import hashlib
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from app import settings


# ============================================================
# Internal Helpers
# ============================================================

def _fb_json_url(path: str, params: dict | None = None) -> str:
    """
    Build Firebase RTDB REST URL:
        https://<db>.firebaseio.com/<path>.json?auth=...
    """
    if not settings.FIREBASE_DB_URL:
        raise RuntimeError("FIREBASE_DB_URL not set")

    path = path.strip("/")

    q = dict(params or {})
    if settings.FIREBASE_DB_AUTH:
        q["auth"] = settings.FIREBASE_DB_AUTH

    if q:
        from urllib.parse import urlencode
        qs = "?" + urlencode(q, safe=":$,()\"")
    else:
        qs = ""

    return f"{settings.FIREBASE_DB_URL}/{path}.json{qs}"


# ============================================================
# Public Firebase Fetch
# ============================================================

def fb_fetch(path: str, params: dict | None = None, timeout: int = 20):
    """
    Fetch JSON from Firebase RTDB.
    """
    url = _fb_json_url(path, params)

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    return r.json()


# ============================================================
# Write PNG to FB_UPLOAD_DIR
# ============================================================

def write_png_to_fb(
    raw_bytes: bytes,
    suggested_name: Optional[str] = None,
) -> str:
    """
    Convert image bytes → PNG and store in FB_UPLOAD_DIR.

    Deduplicates by SHA1 hash.

    Returns:
        filename (string)
    """

    os.makedirs(settings.FB_UPLOAD_DIR, exist_ok=True)

    # Load image
    img = Image.open(io.BytesIO(raw_bytes))

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    # Deduplicate using SHA1
    h = hashlib.sha1(raw_bytes).hexdigest()[:16]

    prefix = ""
    if suggested_name:
        prefix = Path(suggested_name).stem + "_"

    fname = f"{prefix}{h}.png"
    fpath = os.path.join(settings.FB_UPLOAD_DIR, fname)

    if not os.path.exists(fpath):
        with open(fpath, "wb") as f:
            img.save(f, format="PNG")

    return fname