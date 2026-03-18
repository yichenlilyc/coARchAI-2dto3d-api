# app/routers/uploads.py
from __future__ import annotations

import os
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from app import settings
from app.services.storage import safe_join, delete_if_exists

router = APIRouter()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _model_url(filename: str) -> str:
    # served by main.py mount
    rel = f"/static/uploads/models/{filename}"
    return f"{settings.PUBLIC_BASE_URL}{rel}" if settings.PUBLIC_BASE_URL else rel


def _meta_path_for_model(filename: str) -> str:
    stem = Path(filename).stem
    return os.path.join(settings.UPLOAD_MODELS_DIR, f"{stem}.json")


def _preview_filename_for_model(filename: str, preview_ext: str) -> str:
    # preview sits next to model, same stem
    return f"{Path(filename).stem}{preview_ext}"


def _find_existing_preview_url_for_model(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    for ext in settings.ALLOWED_PREVIEW_EXTS:
        cand = os.path.join(settings.UPLOAD_MODELS_DIR, f"{stem}{ext}")
        if os.path.isfile(cand):
            return _model_url(f"{stem}{ext}")
    return None


# ============================================================
# POST /uploads/models  (model + optional preview)
# ============================================================

@router.post("/uploads/models")
async def upload_model(
    file: UploadFile = File(...),
    preview: UploadFile | None = File(None),
):
    """
    Upload a user model (GLB/OBJ) and optionally a preview image (PNG/JPG/WEBP).

    Form-data:
      - file:    required (.glb or .obj)
      - preview: optional (.png/.jpg/.jpeg/.webp)

    Saves:
      - <stem>_<uid>.glb|obj
      - <stem>_<uid>.png|jpg|webp   (if provided)
      - <stem>_<uid>.json           metadata
    Returns:
      Unity-friendly JSON (Option A)
    """
    if not file.filename:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Missing model filename.")

    model_ext = Path(file.filename).suffix.lower()
    if model_ext not in settings.ALLOWED_MODEL_EXTS:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Unsupported model extension: {model_ext}. Allowed: {sorted(settings.ALLOWED_MODEL_EXTS)}",
        )

    # Unique model filename
    stem = Path(file.filename).stem or "model"
    uid = uuid.uuid4().hex[:8]
    model_name = f"{stem}_{uid}{model_ext}"
    model_path = os.path.join(settings.UPLOAD_MODELS_DIR, model_name)

    # Save model
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    preview_url: Optional[str] = None
    preview_name: Optional[str] = None

    # Save preview if provided
    if preview and preview.filename:
        preview_ext = Path(preview.filename).suffix.lower()
        if preview_ext not in settings.ALLOWED_PREVIEW_EXTS:
            raise HTTPException(
                HTTP_400_BAD_REQUEST,
                f"Unsupported preview extension: {preview_ext}. Allowed: {sorted(settings.ALLOWED_PREVIEW_EXTS)}",
            )

        preview_name = _preview_filename_for_model(model_name, preview_ext)
        preview_path = os.path.join(settings.UPLOAD_MODELS_DIR, preview_name)

        with open(preview_path, "wb") as pf:
            shutil.copyfileobj(preview.file, pf)

        preview_url = _model_url(preview_name)

    meta = {
        "name": model_name,
        "url": _model_url(model_name),
        "preview_url": preview_url,   # can be null
        "engine": "upload",
        "mtime": _now_iso(),
        "size": os.stat(model_path).st_size,
    }

    # Write sidecar JSON
    with open(_meta_path_for_model(model_name), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    return meta


# ============================================================
# GET /uploads/models
# ============================================================

@router.get("/uploads/models")
def list_models(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    List uploaded user models.
    Returns Unity-friendly JSON:
      { count, limit, offset, items: [...] }
    """
    items = []

    # Prefer JSON sidecars (stable fields)
    for name in os.listdir(settings.UPLOAD_MODELS_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(settings.UPLOAD_MODELS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
                # ensure preview_url exists (backward compat)
                if not meta.get("preview_url") and meta.get("name"):
                    meta["preview_url"] = _find_existing_preview_url_for_model(meta["name"])
                items.append(meta)
        except Exception:
            continue

    # Fallback: if no JSON sidecars exist, list .glb/.obj directly
    if not items:
        for name in os.listdir(settings.UPLOAD_MODELS_DIR):
            ext = Path(name).suffix.lower()
            if ext not in settings.ALLOWED_MODEL_EXTS:
                continue
            p = os.path.join(settings.UPLOAD_MODELS_DIR, name)
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
            items.append(
                {
                    "name": name,
                    "url": _model_url(name),
                    "preview_url": _find_existing_preview_url_for_model(name),
                    "engine": "upload",
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "size": st.st_size,
                }
            )

    # Sort newest first
    items.sort(key=lambda x: x.get("mtime", ""), reverse=True)

    total = len(items)
    items = items[offset : offset + limit]

    return {"count": total, "limit": limit, "offset": offset, "items": items}


# ============================================================
# DELETE /uploads/models/{name}
# ============================================================

@router.delete("/uploads/models/{name}")
def delete_model(name: str):
    """
    Delete:
      - model file (.glb/.obj)
      - its .json sidecar
      - any preview image with same stem (.png/.jpg/.jpeg/.webp)
    You can pass:
      - full filename (chair_xxx.glb)
      - stem only (chair_xxx)
    """
    stem = Path(name).stem
    if not stem:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Empty model name.")

    deleted = []
    missing = []

    # Delete model(s) for that stem
    model_deleted_any = False
    for ext in settings.ALLOWED_MODEL_EXTS:
        model_name = f"{stem}{ext}"
        p = safe_join(settings.UPLOAD_MODELS_DIR, model_name)
        if delete_if_exists(p):
            deleted.append(model_name)
            model_deleted_any = True
        else:
            missing.append(model_name)

    # Delete JSON sidecar
    json_name = f"{stem}.json"
    jp = safe_join(settings.UPLOAD_MODELS_DIR, json_name)
    if delete_if_exists(jp):
        deleted.append(json_name)
    else:
        missing.append(json_name)

    # Delete preview(s)
    for ext in settings.ALLOWED_PREVIEW_EXTS:
        prev_name = f"{stem}{ext}"
        pp = safe_join(settings.UPLOAD_MODELS_DIR, prev_name)
        if delete_if_exists(pp):
            deleted.append(prev_name)

    if not model_deleted_any:
        # If neither .glb nor .obj existed, treat as not found
        raise HTTPException(HTTP_404_NOT_FOUND, f"Model '{stem}' not found.")

    return {"ok": True, "stem": stem, "deleted": deleted, "missing": missing}


# ============================================================
# DELETE /uploads/models?url=...
# ============================================================

@router.delete("/uploads/models")
def delete_model_by_url(url: str):
    """
    Convenience:
      DELETE /uploads/models?url=/static/uploads/models/chair_xxx.glb
    """
    fname = url.rsplit("/", 1)[-1]
    return delete_model(fname)