# app/services/common.py
from __future__ import annotations

import io
import base64
from typing import Optional

import requests
import trimesh
from PIL import Image
from fastapi.concurrency import run_in_threadpool


# ============================================================
# Image Loading
# ============================================================

def load_image_from_payload(payload: dict) -> Image.Image:
    """
    Accepts:
        {"url": "..."}  OR
        {"b64": "..."}  (raw base64 OR data:image/...;base64,...)

    Returns:
        PIL.Image (RGB)

    Raises:
        ValueError
    """
    if payload.get("url"):
        url = payload["url"]
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            return Image.open(r.raw).convert("RGB")
        except Exception as e:
            raise ValueError(f"URL not accessible: {e}")

    if payload.get("b64"):
        try:
            raw = decode_data_url_to_bytes(payload["b64"])
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise ValueError("Invalid base64 payload.")

    raise ValueError("Missing 'url' or 'b64' in payload.")


# ============================================================
# Async Download Helper
# ============================================================

async def download_bytes(url: str, *, timeout: int = 300) -> bytes:
    """
    Async-safe HTTP download using threadpool.
    """
    def _get() -> bytes:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content

    return await run_in_threadpool(_get)


# ============================================================
# Mesh Helpers
# ============================================================

def mesh_to_glb_bytes(mesh: trimesh.Trimesh) -> bytes:
    """
    Serialize a trimesh mesh to .glb bytes.
    """
    return mesh.export(file_type="glb")


# ============================================================
# Base64 Utilities
# ============================================================

def decode_data_url_to_bytes(data_url: str) -> bytes:
    """
    Accepts:
        data:image/...;base64,...
    OR
        raw base64 string

    Returns:
        decoded bytes
    """
    s = (data_url or "").strip()

    if s.lower().startswith("data:") and ";base64," in s:
        s = s.split(";base64,", 1)[1]

    return base64.b64decode(s)


# ============================================================
# Filename Guessing (Tripo3D helper)
# ============================================================

def guess_filename_from_url(url: str) -> str:
    """
    Guess a reasonable filename based on extension.
    """
    low = (url or "").lower()

    if low.endswith((".jpg", ".jpeg")):
        return "image.jpg"

    if low.endswith(".webp"):
        return "image.webp"

    if low.endswith(".png"):
        return "image.png"

    if low.endswith(".bmp"):
        return "image.bmp"

    return "image.png"