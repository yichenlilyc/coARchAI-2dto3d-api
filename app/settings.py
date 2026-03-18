# app/settings.py
import os
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- GENERATED OUTPUTS ---
GENERATED_IMAGES_DIR = os.getenv("GENERATED_IMAGES_DIR", "/app/generated/images")
GENERATED_MODELS_DIR = os.getenv("GENERATED_MODELS_DIR", "/app/generated/models")
FB_UPLOAD_DIR = os.getenv("FB_UPLOAD_DIR", "/app/fbupload")

for d in (GENERATED_IMAGES_DIR, GENERATED_MODELS_DIR, FB_UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

# --- PRESETS / UPLOADS ---
PRESET_MAPS_DIR = os.getenv("PRESET_MAPS_DIR", "/app/data/presets/maps")
PRESET_MODELS_DIR = os.getenv("PRESET_MODELS_DIR", "/app/data/presets/models")
UPLOAD_MODELS_DIR = os.getenv("UPLOAD_MODELS_DIR", "/app/data/uploads/models")

for d in (PRESET_MAPS_DIR, PRESET_MODELS_DIR, UPLOAD_MODELS_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_PRESET_2D_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_PRESET_3D_EXTS = {".glb",".obj"}

ALLOWED_PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MODEL_EXTS = {".glb", ".obj"}

# --- DEVICE ---
USE_CUDA = bool(int(os.getenv("USE_CUDA", "0")))
CUDA_OK = torch.cuda.is_available() and USE_CUDA
DEVICE = "cuda" if CUDA_OK else "cpu"
DTYPE = torch.float16 if CUDA_OK else torch.float32

# --- SERVER ---
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# --- Firebase RTDB ---
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL", "").rstrip("/")
FIREBASE_DB_AUTH = os.getenv("FIREBASE_DB_AUTH", "")

# --- Tripo3D ---
TRIPO3D_API_KEY = os.getenv("TRIPO3D_API_KEY", "")
USE_TRIPO_SDK = bool(int(os.getenv("USE_TRIPO_SDK", "1")))
TRIPO3D_BASE = os.getenv("TRIPO3D_BASE", "https://api.tripo3d.ai/v2/openapi")
TRIPO3D_MODEL_VERSION = os.getenv("TRIPO3D_MODEL_VERSION", "v2.0-20240919")
TRIPO3D_POLL_SECONDS = float(os.getenv("TRIPO3D_POLL_SECONDS", "2.0"))
TRIPO3D_TIMEOUT_SECONDS = float(os.getenv("TRIPO3D_TIMEOUT_SECONDS", "1800"))

ALLOWED_UPLOAD_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}