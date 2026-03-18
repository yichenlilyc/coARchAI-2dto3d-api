# app/routers/presets.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from app import settings

router = APIRouter()


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext in settings.ALLOWED_PRESET_2D_EXTS:
        return "2d"

    if ext in settings.ALLOWED_PRESET_3D_EXTS:
        return "3d"

    raise HTTPException(
        HTTP_400_BAD_REQUEST,
        f"Unsupported file extension: {ext}"
    )


# ============================================================
# POST /presets/maps
# ============================================================
@router.post("/presets/maps")
async def upload_preset_map(
    location: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload one or multiple files for a location.
    Files can be 2D images OR 3D (.glb, .obj).
    """

    location = location.strip()
    if not location:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Location name required")

    location_dir = Path(settings.PRESET_MAPS_DIR) / location
    location_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    import shutil

    for file in files:
        if not file.filename:
            continue

        _validate_extension(file.filename)

        dest = location_dir / Path(file.filename).name

        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved.append(dest.name)

    return {
        "ok": True,
        "location": location,
        "saved": saved
    }


# ============================================================
# GET /presets/maps
# ============================================================
@router.get("/presets/maps")
def list_presets():
    """
    Returns all locations and their files.
    """

    base = Path(settings.PRESET_MAPS_DIR)

    locations = []

    if not base.exists():
        return {"count": 0, "items": []}

    for loc_dir in base.iterdir():
        if not loc_dir.is_dir():
            continue

        files = []

        for f in loc_dir.iterdir():
            if not f.is_file():
                continue

            ext = f.suffix.lower()

            if ext not in (
                settings.ALLOWED_PRESET_2D_EXTS |
                settings.ALLOWED_PRESET_3D_EXTS
            ):
                continue

            files.append({
                "name": f.name,
                "url": f"/presets/maps/{loc_dir.name}/{f.name}",
                "size": f.stat().st_size,
                "mtime": datetime.fromtimestamp(
                    f.stat().st_mtime
                ).isoformat()
            })

        files.sort(key=lambda x: x["mtime"], reverse=True)

        locations.append({
            "location": loc_dir.name,
            "files": files
        })

    locations.sort(key=lambda x: x["location"])

    return {
        "count": len(locations),
        "items": locations
    }

@router.post("/presets/models")
async def upload_preset_models(
    folder: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    Upload one or multiple primitive models into PRESET_MODELS_DIR.
    Allowed: .glb, .obj

    If folder is provided:
        /presets/models/<folder>/<filename>
    else:
        /presets/models/<filename>
    """
    subdir = (folder or "").strip()
    base_dir = Path(settings.PRESET_MODELS_DIR)

    target_dir = (base_dir / subdir) if subdir else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in settings.ALLOWED_MODEL_EXTS:
            raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported model extension: {ext}")

        dest = target_dir / Path(file.filename).name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved.append(str(dest.relative_to(base_dir)).replace("\\", "/"))

    return {
        "ok": True, 
        "saved": saved, 
        "base": "/presets/models"
    }


@router.get("/presets/models")
def list_preset_models():
    """
    List primitive models under PRESET_MODELS_DIR (recursive).
    Returns URLs that match the StaticFiles mount: /presets/models/<path>
    """
    base_dir = Path(settings.PRESET_MODELS_DIR)
    if not base_dir.exists():
        return {"count": 0, "items": []}

    items = []
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in settings.ALLOWED_MODEL_EXTS:
            continue

        rel = str(p.relative_to(base_dir)).replace("\\", "/")
        items.append({
            "name": p.name,
            "path": rel,
            "url": f"/presets/models/{rel}",
            "size": p.stat().st_size,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        })

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {
        "count": len(items), 
        "items": items
    }