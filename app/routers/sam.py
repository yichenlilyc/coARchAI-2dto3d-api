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
# --- SAM 3 ---
# Lazy Load SAM 3
sam3_model = None
sam3_processor = None
SAM3_LOAD_ERROR: Optional[str] = None

def get_sam3():
    """Lazy-load Meta's official SAM 3 model for both prompts and clicks."""
    global sam3_model, sam3_processor, SAM3_LOAD_ERROR
    if sam3_model is not None and sam3_processor is not None:
        return sam3_model, sam3_processor
    
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        print("Initializing native SAM 3 Model for Prompts and Clicks...")
        sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)
        
        SAM3_LOAD_ERROR = None
        return sam3_model, sam3_processor
    except Exception as e:
        SAM3_LOAD_ERROR = str(e)
        return None, None

def process_sam3_masks(masks):
    """Helper function to safely flatten and combine SAM 3 masks for PIL."""
    if len(masks) == 0:
        raise ValueError("No objects found matching the input.")

    if isinstance(masks, torch.Tensor):
        combined_mask = torch.sum(masks, dim=0).clamp(0, 1)
        mask_uint8 = (combined_mask * 255).cpu().numpy().astype(np.uint8)
    else:
        combined_mask = np.clip(np.sum(masks, axis=0), 0, 1)
        mask_uint8 = (combined_mask * 255).astype(np.uint8)
    
    if mask_uint8.ndim > 2:
        mask_uint8 = np.squeeze(mask_uint8) 
        
    return Image.fromarray(mask_uint8, mode='L')

def run_inference_sam3_prompt_sync(img_pil, text_prompt):
    """Blocking GPU Inference for Text Prompts."""
    model, processor = get_sam3()
    if not model: raise RuntimeError(f"SAM 3 not loaded: {SAM3_LOAD_ERROR}")

    with torch.no_grad():
        inference_state = processor.set_image(img_pil)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
    return process_sam3_masks(output["masks"])

def run_inference_sam3_point_sync(img_pil, x, y):
    """Blocking GPU Inference for Touch/Click Coordinates."""
    model, processor = get_sam3()
    if not model: raise RuntimeError(f"SAM 3 not loaded: {SAM3_LOAD_ERROR}")

    with torch.no_grad():
        # 1. Load image into state
        inference_state = processor.set_image(img_pil)
        
        # 2. Get image dimensions to normalize the coordinates
        img_w, img_h = img_pil.size
        
        # 3. Create a tiny 10x10 pixel box centered on the user's click
        # Using the exact cxcywh format required by Meta's processor
        center_x = x / img_w
        center_y = y / img_h
        
        # Give the box a tiny, non-zero physical area (10 pixels) normalized to percentages
        width = 10.0 / img_w
        height = 10.0 / img_h
        
        box = [center_x, center_y, width, height]
        
        # 5. Pass it to SAM 3's geometric prompter (label=True means positive selection)
        output_state = processor.add_geometric_prompt(
            box=box, 
            label=True, 
            state=inference_state
        )
        
    return process_sam3_masks(output_state["masks"])

# Router Endpoints

@router.post("/image-to-3d/sam/segment-touch")
async def segment_image_point(payload: dict = Body(...)):
    """Handles Touch-to-Mask clicks using SAM 3."""
    try:
        if "image_b64" in payload and "b64" not in payload:
            payload["b64"] = payload["image_b64"]
            
        img = load_image_from_payload(payload)
        x, y = payload.get("x"), payload.get("y")
        
        if x is None or y is None:
            return json_error("Missing x or y coordinates", stage="sam3-point")

        loop = asyncio.get_event_loop()
        mask_img = await loop.run_in_executor(None, run_inference_sam3_point_sync, img, x, y)
        
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        return json_error(str(e), stage="sam3-point", exc=e)

@router.post("/image-to-3d/segment-prompt")
async def segment_image_prompt(payload: dict = Body(...)):
    """Handles Text Prompts using SAM 3."""
    try:
        if "image_b64" in payload and "b64" not in payload:
            payload["b64"] = payload["image_b64"]
            
        img = load_image_from_payload(payload)
        prompt = payload.get("prompt")
        
        if not prompt:
            return json_error("Missing 'prompt' text in payload", stage="sam3-prompt")

        loop = asyncio.get_event_loop()
        mask_img = await loop.run_in_executor(None, run_inference_sam3_prompt_sync, img, prompt)
        
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        return json_error(str(e), stage="sam3-prompt", exc=e)


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