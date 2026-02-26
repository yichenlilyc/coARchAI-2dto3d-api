# app/services/storage.py
from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from app import settings


def safe_join(root_dir: str, filename: str) -> Path:
    if not filename or "/" in filename or "\\" in filename:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Invalid filename.")

    root = Path(root_dir).resolve()
    p = (root / filename).resolve()

    # Robust traversal check (use Path semantics, not string prefix)
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Unsafe path.")

    return p


def delete_if_exists(p: Path) -> bool:
    if p.exists():
        p.unlink()
        return True
    return False


def save_generated_glb(
    glb_bytes: bytes,
    engine: str,
    *,
    source_url: Optional[str] = None,
    seed: Optional[int] = None,
    params: Optional[dict] = None,
) -> dict:
    """
    Save GLB into GENERATED_MODELS_DIR and write a sidecar JSON metadata file.

    IMPORTANT: URL returned uses your static mount:
      /static/generated/models/<name>.glb
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:12]
    base = f"{ts}_{engine}_{uid}"

    glb_name = f"{base}.glb"
    glb_path = os.path.join(settings.GENERATED_MODELS_DIR, glb_name)

    with open(glb_path, "wb") as f:
        f.write(glb_bytes)

    rel = f"/static/generated/models/{glb_name}"
    url = f"{settings.PUBLIC_BASE_URL}{rel}" if settings.PUBLIC_BASE_URL else rel

    meta = {
        "name": glb_name,
        "url": url,
        "engine": engine,
        "source_url": source_url,
        "seed": seed,
        "params": params or {},
        "size": len(glb_bytes),
        "mtime": datetime.utcnow().isoformat() + "Z",
    }

    meta_path = os.path.join(settings.GENERATED_MODELS_DIR, f"{base}.json")
    with open(meta_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    return meta