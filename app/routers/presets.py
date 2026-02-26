# app/routers/presets.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from app import settings
from app.services.storage import safe_join

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


async def _save_upload(file: UploadFile, dest_dir: str):
    import shutil

    if not file.filename:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Missing filename")

    _assert_ext(file.filename, settings.ALLOWED_MODEL_EXTS)

    # keep original name for developer presets (overwrite allowed)
    out_path = safe_join(dest_dir, Path(file.filename).name)

    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return Path(out_path).name


# -----------------------
# Preset Maps
# -----------------------

@router.post("/presets/maps")
async def upload_preset_map(file: UploadFile = File(...)):
    name = await _save_upload(file, settings.PRESET_MAPS_DIR)
    return {"ok": True, "name": name}


@router.get("/presets/maps")
def list_preset_maps():
    return {"count": len(os.listdir(settings.PRESET_MAPS_DIR)), "items": _list_dir(settings.PRESET_MAPS_DIR)}


# -----------------------
# Preset Models
# -----------------------

@router.post("/presets/models")
async def upload_preset_model(file: UploadFile = File(...)):
    name = await _save_upload(file, settings.PRESET_MODELS_DIR)
    return {"ok": True, "name": name}


@router.get("/presets/models")
def list_preset_models():
    return {"count": len(os.listdir(settings.PRESET_MODELS_DIR)), "items": _list_dir(settings.PRESET_MODELS_DIR)}