import sys
import os
import json
import traceback
import uuid
from datetime import datetime
from PIL import Image
import numpy as np
from fastapi import FastAPI, BackgroundTasks
import torch
from pydantic import BaseModel

# Import settings from main app
from app import settings

class TaskPayload(BaseModel):
    task_id: str
    img_path: str
    mask_path: str

# Prevent CUDA from crashing with auto-tune
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "0"

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = "/usr/local/cuda"

# Add the SAM 3D repo to path
SAM3D_REPO_PATH = "external/sam-3d-objects"
NOTEBOOK_PATH = os.path.join(SAM3D_REPO_PATH, "notebook")
sys.path.append(NOTEBOOK_PATH)

from inference import Inference

app = FastAPI(title="SAM 3D Internal Worker API")

pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    print("Loading SAM 3D model onto GPU 1...")
    possible_paths = [
        "/app/checkpoints/hf/checkpoints/pipeline.yaml",
        "/app/checkpoints/pipeline.yaml",
        "external/sam-3d-objects/checkpoints/pipeline.yaml"
    ]

    pipeline_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not pipeline_path:
        raise FileNotFoundError("Could not find SAM-3D pipeline.yaml in expected locations.")
    
    pipeline = Inference(pipeline_path, compile=False)
    print("SAM 3D model loaded successfully.")

def update_ticket(task_id: str, status: str, error: str = None, **kwargs):
    """Updates the JSON ticket so the Main API can poll the status."""
    ticket_path = os.path.join(settings.SAM_TASKS_DIR, f"{task_id}.json")
    
    # read existing data first
    data = {}
    if os.path.exists(ticket_path):
        try:
            with open(ticket_path, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    # update with new status and kwargs
    data["task_id"] = task_id
    data["status"] = status
    if error: data["error"] = error
    data.update(kwargs)

    temp_path = f"{ticket_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f)
    os.replace(temp_path, ticket_path)

def smart_crop(img_rgb, mask_uint8, margin=0.1):
    coords = np.argwhere(mask_uint8 > 128)
    if coords.size == 0:
        return img_rgb, mask_uint8

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    h, w = y1 - y0, x1 - x0
    cy, cx = y0 + h // 2, x0 + w // 2
    size = int(max(h, w) * (1 + margin))
    
    left, top = max(0, cx - size // 2), max(0, cy - size // 2)
    right, bottom = min(img_rgb.shape[1], cx + size // 2), min(img_rgb.shape[0], cy + size // 2)

    return img_rgb[top:bottom, left:right], mask_uint8[top:bottom, left:right]

def run_3d_generation(task_id: str, img_path: str, mask_path: str):
    """The core generation logic, outputting directly to the gallery directory."""
    update_ticket(task_id, "processing")
    try:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))
        mask = (mask_raw > 128).astype(np.uint8) * 255

        if mask.max() == 0: raise ValueError("Mask is empty. Please provide a valid segmentation mask.")
        img_rgb, mask = smart_crop(img_rgb, mask)
        if mask.max() == 0: raise ValueError("After cropping, mask is empty.")

        print(f"Running 3D generation for task {task_id}...")
        
        # Force the pipeline to generate meshes even if previous runs set it to False
        for obj in [pipeline, getattr(pipeline, 'pipeline', None), getattr(pipeline, 'model', None)]:
            if obj is not None:
                if hasattr(obj, 'with_mesh_postprocess'): obj.with_mesh_postprocess = True
                if hasattr(obj, 'with_texture_baking'): obj.with_texture_baking = True

        output = pipeline(img_rgb, mask, seed=42)

        # gallery info
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        uid = uuid.uuid4().hex[:12]
        base_name = f"{ts}_SAM-3D_{uid}"

        glb_name = f"{base_name}.glb"
        ply_name = f"{base_name}.ply"

        glb_path = os.path.join(settings.GENERATED_MODELS_DIR, glb_name)
        ply_path = os.path.join(settings.GENERATED_MODELS_DIR, ply_name)

        # Save files natively to the app's models directory
        output["gs"].save_ply(ply_path)
        output["glb"].export(glb_path)

        # Build Sidecar JSON for the frontend gallery
        rel = f"/static/generated/models/{glb_name}"
        url = f"{settings.PUBLIC_BASE_URL}{rel}" if settings.PUBLIC_BASE_URL else rel

        meta = {
            "name": glb_name,
            "url": url,
            "engine": "SAM-3D",
            "source_url": None,
            "seed": 42,
            "params": {"task_id": task_id},
            "size": os.path.getsize(glb_path),
            "mtime": datetime.utcnow().isoformat() + "Z",
        }

        meta_path = os.path.join(settings.GENERATED_MODELS_DIR, f"{base_name}.json")
        with open(meta_path, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=2)

        print(f"Task {task_id} completed. Saved as {glb_name} in Gallery.")
        
        # Give the main API the exact paths to serve
        update_ticket(task_id, "completed", glb_file=glb_path, ply_file=ply_path)

    except Exception as e:
        traceback.print_exc()
        update_ticket(task_id, "failed", error=str(e))
    finally:
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(mask_path): os.remove(mask_path)

@app.post("/process-3d")
async def process_3d(payload: TaskPayload, background_tasks: BackgroundTasks):
    """Receives the job from the main API and processes it in the background."""
    print(f"Worker received task: {payload.task_id}")
    background_tasks.add_task(run_3d_generation, payload.task_id, payload.img_path, payload.mask_path)
    return {"status": "accepted"}