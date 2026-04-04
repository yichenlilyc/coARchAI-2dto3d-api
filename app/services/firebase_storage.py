from datetime import datetime
from app.database import get_db, get_bucket

def save_model_to_firebase(file_bytes: bytes, user_id: str, model_id: str, source_image_id: str = "unknown", format: str = "glb"):
    """
    Uploads the 3D mesh or splat to Firebase Storage and merges metadata into Firestore.
    """
    bucket = get_bucket()
    db = get_db()
    
    if not bucket or not db:
        raise Exception("Firebase is not initialized. Check your service account credentials.")

    # Use the provided model_id (which will be the task_id)
    file_path = f"models/{user_id}/{model_id}.{format}"
    
    # Upload with the correct MIME type
    content_type = "model/gltf-binary" if format == "glb" else "application/octet-stream"
    blob = bucket.blob(file_path)
    blob.upload_from_string(file_bytes, content_type=content_type)
    
    # Make public
    blob.make_public()
    public_url = blob.public_url
    
    # Save metadata USING MERGE=TRUE
    doc_data = {
        "model_id": model_id,
        "user_id": user_id,
        "source_image_id": source_image_id,
        f"{format}_url": public_url, # Dynamically saves as 'glb_url' OR 'ply_url'
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "completed",
    }
    
    # merge true adds the PLY link to the existing GLB document
    db.collection("models_3d").document(model_id).set(doc_data, merge=True)
    
    return doc_data