import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from firebase_admin import firestore

from app.database import get_db, get_bucket

router = APIRouter()

# 2D Image Endpoints

@router.post("/generated/images")
async def upload_image(
    file: UploadFile = File(...), 
    user_id: str = Form("anonymous_student") # Accepts user_id from the frontend form
):
    """Uploads a 2D image to Firebase Storage and saves metadata to Firestore."""
    db, bucket = get_db(), get_bucket()
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase not connected.")

    # 1. Read file and generate IDs
    file_bytes = await file.read()
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else "png"
    file_path = f"images/{user_id}/{image_id}.{ext}"

    # 2. Upload to Firebase Storage
    blob = bucket.blob(file_path)
    blob.upload_from_string(file_bytes, content_type=file.content_type)
    blob.make_public()

    # 3. Save to Firestore
    doc_data = {
        "image_id": image_id,
        "user_id": user_id,
        "storage_url": blob.public_url,
        "filename": file.filename,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "size_bytes": len(file_bytes)
    }
    db.collection("images_2d").document(image_id).set(doc_data)

    return {"status": "success", "image_data": doc_data}

@router.get("/generated/images")
def list_images(user_id: Optional[str] = None, limit: int = 100):
    """Fetches the gallery of 2D images directly from Firestore."""
    db = get_db()
    
    images_ref = db.collection("images_2d")
    if user_id:
        query = images_ref.where("user_id", "==", user_id).order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
    else:
        query = images_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
        
    results = query.stream()
    items = [doc.to_dict() for doc in results]
        
    return {"count": len(items), "items": items}


# 3D Model Endpoints

@router.post("/generated/models")
async def upload_model_manual(
    file: UploadFile = File(...), 
    user_id: str = Form("anonymous_student")
):
    """Allows students to manually upload a .glb file to their Firebase gallery."""
    db, bucket = get_db(), get_bucket()

    file_bytes = await file.read()
    model_id = f"mod_{uuid.uuid4().hex[:12]}"
    file_path = f"models/{user_id}/{model_id}.glb"

    # Upload to Storage
    blob = bucket.blob(file_path)
    blob.upload_from_string(file_bytes, content_type="model/gltf-binary")
    blob.make_public()

    # Save to Firestore
    doc_data = {
        "model_id": model_id,
        "user_id": user_id,
        "source_image_id": "manual_upload",
        "glb_url": blob.public_url,
        "filename": file.filename,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "completed",
        "size_bytes": len(file_bytes)
    }
    db.collection("models_3d").document(model_id).set(doc_data)

    return {"status": "success", "model_data": doc_data}

@router.get("/generated/models")
def list_models(user_id: Optional[str] = None, limit: int = 100):
    """Fetches the gallery of 3D models directly from Firestore."""
    db = get_db()
    
    models_ref = db.collection("models_3d")
    if user_id:
        query = models_ref.where("user_id", "==", user_id).order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
    else:
        query = models_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
        
    results = query.stream()
    items = [doc.to_dict() for doc in results]
        
    return {"count": len(items), "items": items}

# Deletion Endpoints

@router.delete("/generated/images/{image_id}")
def delete_image(
    image_id: str, 
    user_id: str = Query(..., description="The ID of the user requesting deletion")
):
    """Deletes the 2D image from Firebase Storage and removes its Firestore record."""
    db, bucket = get_db(), get_bucket()
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase not connected.")

    # Fetch the database record to verify ownership and get file details
    doc_ref = db.collection("images_2d").document(image_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Image record not found.")
        
    data = doc.to_dict()
    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized to delete this image.")

    # Reconstruct the file path and delete from Storage
    ext = data.get("filename", "").split(".")[-1].lower() if "." in data.get("filename", "") else "png"
    file_path = f"images/{user_id}/{image_id}.{ext}"
    
    try:
        blob = bucket.blob(file_path)
        if blob.exists():
            blob.delete()
    except Exception as e:
        print(f"Warning: Could not delete storage blob {file_path}: {e}")

    # Delete the record from Firestore
    doc_ref.delete()

    return {"status": "success", "deleted_image_id": image_id}


@router.delete("/generated/models/{model_id}")
def delete_model(
    model_id: str, 
    user_id: str = Query(..., description="The ID of the user requesting deletion")
):
    """Deletes both the 3D mesh (.glb) and splat (.ply) from Firebase Storage and removes its Firestore record."""
    db, bucket = get_db(), get_bucket()
    if not db or not bucket:
        raise HTTPException(status_code=500, detail="Firebase not connected.")

    # Fetch the database record to verify ownership
    doc_ref = db.collection("models_3d").document(model_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Model record not found.")
        
    data = doc.to_dict()
    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized to delete this model.")

    # Reconstruct the file paths and delete BOTH from Storage
    formats_to_delete = ["glb", "ply"]
    
    for fmt in formats_to_delete:
        file_path = f"models/{user_id}/{model_id}.{fmt}"
        try:
            blob = bucket.blob(file_path)
            if blob.exists():
                blob.delete()
                print(f"Successfully deleted: {file_path}")
        except Exception as e:
            print(f"Warning: Could not delete storage blob {file_path}: {e}")

    # Delete the single combined record from Firestore
    doc_ref.delete()

    return {"status": "success", "deleted_model_id": model_id, "formats_cleaned": formats_to_delete}