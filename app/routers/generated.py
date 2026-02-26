# app/routers/generated.py
from __future__ import annotations

import os
import json
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query

from fastapi.responses import FileResponse
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST

from app import settings
from app.services.storage import safe_join, delete_if_exists

router = APIRouter()


# =======================
# Generated Images
# =======================

@router.post("/generated/images")
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts multipart form-data 'file' and saves it under GENERATED_IMAGES_DIR.
    Returns a relative URL you can hand back into /image-to-3d/* endpoints.
    """
    import uuid, shutil

    ext = ""
    if file.filename and "." in file.filename:
        ext = "." + file.filename.split(".")[-1].lower()

    fname = f"{uuid.uuid4().hex}{ext or '.png'}"
    out_path = os.path.join(settings.GENERATED_IMAGES_DIR, fname)

    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"url": f"/static/generated/images/{fname}"}


@router.get("/generated/images")
def list_uploads(limit: int = 100, offset: int = 0):
    items = []
    for name in os.listdir(settings.GENERATED_IMAGES_DIR):
        if name.startswith("."):
            continue
        path = os.path.join(settings.GENERATED_IMAGES_DIR, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        rel = f"/static/generated/images/{name}"
        url = f"{settings.PUBLIC_BASE_URL}{rel}" if settings.PUBLIC_BASE_URL else rel
        items.append(
            {
                "name": name,
                "url": url,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )

    items.sort(key=lambda x: x["mtime"], reverse=True)
    total = len(items)
    items = items[offset : offset + limit]
    return {"count": total, "items": items, "limit": limit, "offset": offset}


@router.delete("/generated/images/{filename}")
def delete_upload(filename: str):
    """
    Delete a single uploaded image from /static/generated/images.
    Only allows typical image extensions.
    """
    p = safe_join(settings.GENERATED_IMAGES_DIR, filename)
    ext = p.suffix.lower()
    if ext not in settings.ALLOWED_UPLOAD_EXTS:
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported file extension: {ext}")

    if delete_if_exists(p):
        return {"ok": True, "deleted": filename}
    raise HTTPException(HTTP_404_NOT_FOUND, f"Upload '{filename}' not found.")


@router.delete("/generated/images")
def delete_upload_by_url(url: str):
    """
    Convenience: DELETE /generated/images?url=/generated/images/abc.png
    """
    fname = url.rsplit("/", 1)[-1]
    return delete_upload(fname)


# =======================
# Generated Models
# =======================

@router.get("/generated/models")
def list_models(limit: int = 100, offset: int = 0):
    items = []
    for name in os.listdir(settings.GENERATED_MODELS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(settings.GENERATED_MODELS_DIR, name), "r", encoding="utf-8") as jf:
                meta = json.load(jf)
                items.append(meta)
        except Exception:
            stem = name[:-5]
            glb = stem + ".glb"
            glb_path = os.path.join(settings.GENERATED_MODELS_DIR, glb)
            if os.path.isfile(glb_path):
                st = os.stat(glb_path)
                items.append(
                    {
                        "name": glb,
                        "url": f"/static/generated/models/{glb}",
                        "engine": "unknown",
                        "source_url": None,
                        "seed": None,
                        "params": {},
                        "size": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    }
                )

    items.sort(key=lambda x: x.get("mtime", ""), reverse=True)
    total = len(items)
    items = items[offset : offset + limit]
    return {"count": total, "items": items, "limit": limit, "offset": offset}


@router.post("/generated/models")
async def save_model(file: UploadFile = File(...)):
    """
    Accepts multipart form-data 'file' (GLB) and saves into GENERATED_MODELS_DIR.
    Returns the saved URL + name.
    """
    import uuid, shutil

    name = file.filename or "model.glb"
    if not name.lower().endswith(".glb"):
        name = name + ".glb"

    stem = Path(name).stem
    fname = f"{stem}_{uuid.uuid4().hex[:8]}.glb"
    out_path = os.path.join(settings.GENERATED_MODELS_DIR, fname)

    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    rel = f"/static/generated/models/{fname}"
    url = f"{settings.PUBLIC_BASE_URL}{rel}" if settings.PUBLIC_BASE_URL else rel

    meta = {
        "name": fname,
        "url": url,
        "engine": "upload",
        "source_url": None,
        "seed": None,
        "params": {},
        "size": os.stat(out_path).st_size,
        "mtime": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(settings.GENERATED_MODELS_DIR, f"{Path(fname).stem}.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    return {"url": url, "name": fname}


@router.get("/generated/models/archive")
def download_models_zip(
    background_tasks: BackgroundTasks,
    engine: Optional[str] = Query(None, description="Filter by engine name (e.g., Tripo3D, TripoSR, ShapE)"),
    since: Optional[str] = Query(None, description="ISO timestamp (UTC) — include files modified AT or AFTER this time"),
    limit: Optional[int] = Query(None, description="Max number of most-recent models to include"),
):
    """
    Create a ZIP containing generated GLB models from /generated/models.
    Filters:
      - engine: only include models whose sidecar JSON has matching `engine`
      - since:  ISO UTC like '2025-10-31T00:00:00Z'
      - limit:  cap count after sorting by mtime desc
    """
    items = []
    for name in os.listdir(settings.GENERATED_MODELS_DIR):
        if not name.endswith(".json"):
            continue
        meta_path = os.path.join(settings.GENERATED_MODELS_DIR, name)
        try:
            with open(meta_path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
        except Exception:
            continue

        glb_name = Path(name).stem + ".glb"
        glb_path = os.path.join(settings.GENERATED_MODELS_DIR, glb_name)
        if not os.path.isfile(glb_path):
            continue

        if engine and str(meta.get("engine", "")).lower() != engine.lower():
            continue

        mt = meta.get("mtime")
        try:
            mt_dt = datetime.fromisoformat(mt.replace("Z", "+00:00")) if mt else None
        except Exception:
            mt_dt = None

        items.append(
            {
                "glb_name": glb_name,
                "glb_path": glb_path,
                "mtime": mt_dt or datetime.utcfromtimestamp(os.path.getmtime(glb_path)),
            }
        )

    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            items = [it for it in items if it["mtime"] >= since_dt]
        except Exception:
            pass

    items.sort(key=lambda it: it["mtime"], reverse=True)

    if isinstance(limit, int) and limit and limit > 0:
        items = items[:limit]

    if not items:
        raise HTTPException(404, "No models match the criteria.")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=f"_models_{ts}.zip")
    tmp_zip_path = tmp_zip.name
    tmp_zip.close()

    with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for it in items:
            zf.write(it["glb_path"], arcname=it["glb_name"])
            meta_name = Path(it["glb_name"]).stem + ".json"
            meta_path = os.path.join(settings.GENERATED_MODELS_DIR, meta_name)
            if os.path.isfile(meta_path):
                zf.write(meta_path, arcname=meta_name)

    background_tasks.add_task(os.remove, tmp_zip_path)

    headers = {"Cache-Control": "no-store"}
    return FileResponse(
        tmp_zip_path,
        media_type="application/zip",
        filename=f"models_{ts}.zip",
        headers=headers,
        background=background_tasks,
    )


@router.delete("/generated/models/{name}")
def delete_model(name: str):
    """
    Delete a saved model by name. You can pass:
      - exact GLB filename (e.g., 20251103T010203Z_Tripo3D_abcd1234.glb)
      - exact JSON sidecar filename (... .json)
      - the stem only (e.g., 20251103T010203Z_Tripo3D_abcd1234)

    Returns which files were deleted/missing.
    """
    stem = Path(name).stem
    if not stem:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Empty model name.")

    glb_name = f"{stem}.glb"
    json_name = f"{stem}.json"

    glb_path = safe_join(settings.GENERATED_MODELS_DIR, glb_name)
    json_path = safe_join(settings.GENERATED_MODELS_DIR, json_name)

    deleted, missing = [], []

    if delete_if_exists(glb_path):
        deleted.append(glb_name)
    else:
        missing.append(glb_name)

    if delete_if_exists(json_path):
        deleted.append(json_name)
    else:
        missing.append(json_name)

    if len(deleted) == 0 and len(missing) > 0:
        raise HTTPException(HTTP_404_NOT_FOUND, f"Model '{stem}' not found.")

    return {"ok": True, "stem": stem, "deleted": deleted, "missing": missing}


@router.delete("/generated/models")
def delete_model_by_url(url: str):
    """
    Convenience: DELETE /generated/models?url=/generated/models/2025...abcd.glb
    """
    fname = url.rsplit("/", 1)[-1]
    return delete_model(fname)