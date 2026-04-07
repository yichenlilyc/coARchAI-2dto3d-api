# app/routers/uploads.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from firebase_admin import firestore

from app import settings
from app.database import get_db, get_bucket 

router = APIRouter()

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

# POST /uploads/models  (model and preview)
@router.post("/uploads/models")
async def upload_model(
    file: UploadFile = File(...),
    preview: UploadFile | None = File(None),
    user_id: str = Form("anonymous_student") # need user id from frontend
):
    """
    Uploads a user model and optional preview image to Firebase Storage.
    """
    db, bucket = get_db(), get_bucket()
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase not connected.")

    if not file.filename:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Missing model filename.")

    model_ext = Path(file.filename).suffix.lower()
    if model_ext not in settings.ALLOWED_MODEL_EXTS:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Unsupported model extension: {model_ext}. Allowed: {sorted(settings.ALLOWED_MODEL_EXTS)}",
        )

    # Generate IDs
    stem = Path(file.filename).stem or "model"
    model_id = f"mod_{uuid.uuid4().hex[:12]}"
    
    # UPLOAD 3D MODEL TO FIREBASE
    file_bytes = await file.read()
    model_file_path = f"models/{user_id}/{model_id}{model_ext}"
    
    model_blob = bucket.blob(model_file_path)
    model_blob.upload_from_string(file_bytes, content_type="model/gltf-binary")
    model_blob.make_public()

    preview_url: Optional[str] = None

    # UPLOAD PREVIEW IMAGE TO FIREBASE 
    if preview and preview.filename:
        preview_ext = Path(preview.filename).suffix.lower()
        if preview_ext not in settings.ALLOWED_PREVIEW_EXTS:
            raise HTTPException(
                HTTP_400_BAD_REQUEST,
                f"Unsupported preview extension: {preview_ext}."
            )

        preview_bytes = await preview.read()
        preview_file_path = f"models/{user_id}/{model_id}_preview{preview_ext}"
        
        preview_blob = bucket.blob(preview_file_path)
        preview_blob.upload_from_string(preview_bytes, content_type=preview.content_type)
        preview_blob.make_public()
        
        preview_url = preview_blob.public_url

    # SAVE METADATA TO FIRESTORE (files should not be saved locally on server)
    doc_data = {
        "model_id": model_id,
        "user_id": user_id,
        "name": f"{stem}{model_ext}",
        "url": model_blob.public_url,          # GLB URL
        "preview_url": preview_url,            # Thumbnail URL
        "engine": "upload",
        "mtime": _now_iso(),
        "size": len(file_bytes),
    }
    
    db.collection("models_3d").document(model_id).set(doc_data)

    return {"status": "success", "model_data": doc_data}



# GET /uploads/models

@router.get("/uploads/models")
def list_models(
    user_id: Optional[str] = None, 
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Fetches uploaded user models directly from Firestore.
    """
    db = get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Firebase not connected.")
        
    models_ref = db.collection("models_3d").where("engine", "==", "upload")
    
    if user_id:
        query = models_ref.where("user_id", "==", user_id).order_by("mtime", direction=firestore.Query.DESCENDING).limit(limit)
    else:
        query = models_ref.order_by("mtime", direction=firestore.Query.DESCENDING).limit(limit)
        
    results = query.stream()
    items = [doc.to_dict() for doc in results]
        
    return {"count": len(items), "items": items}



# DELETE /uploads/models/{model_id}

@router.delete("/uploads/models/{model_id}")
def delete_model(
    model_id: str, 
    user_id: str = Query(..., description="The ID of the user requesting deletion")
):
    """
    Deletes the model and its preview image from Firebase Storage and Firestore.
    """
    db, bucket = get_db(), get_bucket()
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase not connected.")

    # verify ownership
    doc_ref = db.collection("models_3d").document(model_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Model record not found.")
        
    data = doc.to_dict()
    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized to delete this model.")

    # try to delete all possible formats/previews associated with this ID
    formats_to_delete = [".glb", ".obj", "_preview.png", "_preview.jpg", "_preview.jpeg", "_preview.webp"]
    deleted_files = []
    
    for fmt in formats_to_delete:
        file_path = f"models/{user_id}/{model_id}{fmt}"
        try:
            blob = bucket.blob(file_path)
            if blob.exists():
                blob.delete()
                deleted_files.append(file_path)
        except Exception as e:
            print(f"Warning: Could not delete storage blob {file_path}: {e}")

    # delete the Firestore record
    doc_ref.delete()

    return {"ok": True, "deleted_model_id": model_id, "files_cleaned": deleted_files}