from __future__ import annotations

import os
import io
import json
import uuid
import base64
import asyncio
import requests
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import numpy as np
import torch

# Import App Sevices
from app import settings
from app.services.common import load_image_from_payload, decode_data_url_to_bytes
from app.services.errors import json_error
from app.services.firebase_storage import save_model_to_firebase

router = APIRouter()
# --- SAM 2 ---
# Lazy Load SAM 2
sam2_model = None
SAM2_LOAD_ERROR: Optional[str] = None

def get_sam2_predictor():
    """Lazy-load SAM 2 once, reuse across requests."""
    global sam2_model, SAM2_LOAD_ERROR
    if sam2_model is not None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(sam2_model)
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        CHECKPOINT_PATH = "/app/checkpoints/sam2.1_hiera_large.pt"
        CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Missing SAM 2 checkpoint at {CHECKPOINT_PATH}")
            
        sam2_model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=settings.DEVICE)
        SAM2_LOAD_ERROR = None
        return SAM2ImagePredictor(sam2_model)
    except Exception as e:
        SAM2_LOAD_ERROR = str(e)
        return None

def run_inference_sync(img_rgb, x, y):
    """Blocking GPU Inference for SAM 2."""
    predictor = get_sam2_predictor()
    if not predictor:
        raise RuntimeError(f"SAM 2 not loaded: {SAM2_LOAD_ERROR}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        img_np = np.array(img_rgb)
        predictor.set_image(img_np)
        
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]), 
            multimask_output=True
        )
    
    # Grab the highest confidence mask
    best_mask = masks[np.argmax(scores)]
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode='L')


# 2D Segmentation (SAM 2)
@router.post("/image-to-3d/sam/segment")
async def segment_image(payload: dict = Body(...)):
    try:
        # Map frontend 'image_b64' to standard 'b64' for common.py compatibility
        if "image_b64" in payload and "b64" not in payload:
            payload["b64"] = payload["image_b64"]
            
        img = load_image_from_payload(payload)
        x, y = payload.get("x"), payload.get("y")
        
        if x is None or y is None:
            return json_error("Missing x or y coordinates in payload", stage="sam-segment")

        loop = asyncio.get_event_loop()
        mask_img = await loop.run_in_executor(None, run_inference_sync, img, x, y)
        
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        return json_error(str(e), stage="sam-segment", exc=e)
    
# --- SAM 3 ---
# Lazy Load SAM 3
sam3_model = None
sam3_processor = None
SAM3_LOAD_ERROR: Optional[str] = None

def get_sam3():
    """Lazy-load SAM 3 once for text-based prompt segmentation."""
    global sam3_model, sam3_processor, SAM3_LOAD_ERROR
    if sam3_model is not None and sam3_processor is not None:
        return sam3_model, sam3_processor
    
    try:
        from transformers import Sam3Processor, Sam3Model
        
        # SAM 3 requires a HuggingFace login/token to access 'facebook/sam3'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
        SAM3_LOAD_ERROR = None
        return sam3_model, sam3_processor
    except Exception as e:
        SAM3_LOAD_ERROR = str(e)
        return None, None

def run_inference_sam3_sync(img_rgb, text_prompt):
    """Blocking GPU Inference for SAM 3 Text Prompts."""
    model, processor = get_sam3()
    if not model:
        raise RuntimeError(f"SAM 3 not loaded: {SAM3_LOAD_ERROR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process text and image directly through the SAM 3 Processor
    inputs = processor(images=img_rgb, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    masks = results["masks"]
    if len(masks) == 0:
        raise ValueError(f"No objects found matching the prompt: '{text_prompt}'")
        
    # If the text prompt matches multiple objects (e.g., "windows"), combine them into one mask
    combined_mask = torch.sum(masks, dim=0).clamp(0, 1)
    mask_uint8 = (combined_mask * 255).cpu().numpy().astype(np.uint8)
    
    return Image.fromarray(mask_uint8, mode='L')

# SAM 3 Text Prompt Segmentation
@router.post("/image-to-3d/sam3/segment-prompt")
async def segment_image_prompt_sam3(payload: dict = Body(...)):
    """Accepts a base64 image and a 'prompt' string. Returns a combined segmentation mask."""
    try:
        if "image_b64" in payload and "b64" not in payload:
            payload["b64"] = payload["image_b64"]
            
        img = load_image_from_payload(payload)
        prompt = payload.get("prompt")
        
        if not prompt:
            return json_error("Missing 'prompt' text in payload", stage="sam3-segment")

        loop = asyncio.get_event_loop()
        mask_img = await loop.run_in_executor(None, run_inference_sam3_sync, img, prompt)
        
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        return json_error(str(e), stage="sam3-segment", exc=e)


# Start 3D Generation (SAM 3D Worker)
@router.post("/image-to-3d/sam/generate")
async def generate_3d_sam(payload: dict = Body(...)):
    task_id = str(uuid.uuid4())

    try: 
        if "image_b64" in payload and "b64" not in payload:
            payload["b64"] = payload["image_b64"]
            
        img = load_image_from_payload(payload)
        
        if "mask_b64" not in payload:
            return json_error("Missing 'mask_b64' in payload", stage="sam-generate")
            
        # Decode mask using your existing helper
        mask_raw = decode_data_url_to_bytes(payload["mask_b64"])
        mask_img = Image.open(io.BytesIO(mask_raw)).convert("L")

        img_path = os.path.join(settings.SAM_TEMP_DIR, f"{task_id}_image.png")
        mask_path = os.path.join(settings.SAM_TEMP_DIR, f"{task_id}_mask.png")

        img.save(img_path)
        mask_img.save(mask_path)

        ticket_path = os.path.join(settings.SAM_TASKS_DIR, f"{task_id}.json")
        with open(ticket_path, "w") as f:
            json.dump({"task_id": task_id, "status": "queued"}, f)

        # Handoff to GPU 1 Worker
        worker_payload = {"task_id": task_id, "img_path": img_path, "mask_path": mask_path}
        
        try:
            # Ping the URL defined in settings.py
            response = requests.post(settings.SAM_WORKER_URL, json=worker_payload, timeout=3.0)
            response.raise_for_status() 
        except Exception as comm_error:
            with open(ticket_path, "w") as f:
                json.dump({"task_id": task_id, "status": "failed", "error": "Internal Worker offline"}, f)
            return json_error("Internal 3D Worker is offline.", stage="sam-handoff", exc=comm_error)

        return {"task_id": task_id, "status": "queued"}
    
    except Exception as e:
        return json_error("Failed to queue SAM 3D generation", stage="sam-generate", exc=e)


# Poll Status & Serve File
@router.get("/image-to-3d/sam/status/{task_id}")
async def check_sam_status(task_id: str, format: str = "glb", user_id: str = "anonymous_student"):
    if format not in ["glb", "ply"]:
        return json_error("Invalid format. Use 'glb' or 'ply'.", stage="sam-status")
    
    status_file = os.path.join(settings.SAM_TASKS_DIR, f"{task_id}.json")
    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Task not found")
    
    try: 
        with open(status_file, "r") as f:
            data = json.load(f)
        
        if data.get("status") == "completed":
            target_file = data.get(f"{format}_file") 
            
            if target_file and os.path.exists(target_file):
                # Check if we already uploaded it
                firebase_key = f"firebase_{format}_url"
                
                if firebase_key not in data:
                    # Read the local mesh file
                    with open(target_file, "rb") as mesh_file:
                        file_bytes = mesh_file.read()
                    
                    # Upload to Firebase and get the public URL
                    firebase_record = save_model_to_firebase(
                        file_bytes=file_bytes, 
                        user_id=user_id, 
                        model_id=task_id,
                        source_image_id=data.get("source_image_id", "unknown"),
                        format=format
                    )
                    
                    # Save the new Firebase URL into the local so we don't upload it twice
                    data[firebase_key] = firebase_record[f"{format}_url"]
                    with open(status_file, "w") as f:
                        json.dump(data, f)
                
                # Instead of saving the heavy file locally, redirect the user to download it from firebase
                return JSONResponse(content={
                    "status": "completed", 
                    "format": format,
                    "download_url": data[firebase_key]
                })

            return JSONResponse(content={"status": "error", "error": f"{format.upper()} file missing on server."})
            
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        return JSONResponse(content={"status": "processing"})