# app/routers/uploads.py
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from app import settings
from app.services.storage import safe_join, delete_if_exists

router = APIRouter()


def _assert_ext(filename: str, allowed: set[str]):
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported file extension: {ext}")
    return ext


def _list_dir(dir_path: str):
    items = []
    for name in os.listdir(dir_path):
        if name.startswith("."):
            continue
        p = os.path.join(dir_path, name)
        if not os.path.isfile(p):
            continue
        st = os.stat(p)
        items.append(
            {
                "name": name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
            }
        )
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


@router.post("/uploads/models")
async def upload_user_model(file: UploadFile = File(...)):
    import shutil

    if not file.filename:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Missing filename")

    _assert_ext(file.filename, settings.ALLOWED_MODEL_EXTS)

    # user uploads should be unique and never overwrite
    stem = Path(file.filename).stem
    ext = Path(file.filename).suffix.lower()
    fname = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"

    out_path = os.path.join(settings.UPLOAD_MODELS_DIR, fname)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"ok": True, "name": fname}


@router.get("/uploads/models")
def list_user_models():
    return {"count": len(os.listdir(settings.UPLOAD_MODELS_DIR)), "items": _list_dir(settings.UPLOAD_MODELS_DIR)}


@router.delete("/uploads/models/{name}")
def delete_user_model(name: str):
    p = safe_join(settings.UPLOAD_MODELS_DIR, name)

    if not p.exists():
        raise HTTPException(HTTP_404_NOT_FOUND, f"Upload '{name}' not found.")

    if delete_if_exists(p):
        return {"ok": True, "deleted": name}

    raise HTTPException(500, f"Failed to delete '{name}'")


@router.delete("/uploads/models")
def delete_all_user_models(confirm: bool = Query(False, description="Must be true to delete everything")):
    if confirm is not True:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Set confirm=true to delete all uploads.")

    deleted = []
    for name in os.listdir(settings.UPLOAD_MODELS_DIR):
        if name.startswith("."):
            continue
        p = safe_join(settings.UPLOAD_MODELS_DIR, name)
        if p.exists() and p.is_file():
            if delete_if_exists(p):
                deleted.append(name)

    return {"ok": True, "deleted": deleted, "count": len(deleted)}