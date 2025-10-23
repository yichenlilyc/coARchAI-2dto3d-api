# 2dto3d-server: Simple FastAPI server for 2D→3D using Shap-E and TripoSR
# Run inside the venv:
#   .\.venv-2dto3d\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000

import io
import os
import sys
import base64
import traceback
import tempfile
import importlib
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
import requests
import trimesh

import asyncio
import time
import json
import hashlib
from urllib.parse import urlencode, quote

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from datetime import datetime


# --- CONFIG / DEVICE ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/upload")
os.makedirs(UPLOAD_DIR, exist_ok=True)
USE_CUDA = bool(int(os.getenv("USE_CUDA", "0")))  # default 0 (CPU); set 1 to prefer CUDA if available
CUDA_OK = torch.cuda.is_available() and USE_CUDA
DEVICE = "cuda" if CUDA_OK else "cpu"
DTYPE = torch.float16 if CUDA_OK else torch.float32

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# --- Firebase RTDB (REST) config ---
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL", "").rstrip("/")
FIREBASE_DB_AUTH = os.getenv("FIREBASE_DB_AUTH", "")
FB_UPLOAD_DIR = os.getenv("FB_UPLOAD_DIR", "/app/fbupload")
os.makedirs(FB_UPLOAD_DIR, exist_ok=True)

# --- Tripo3D (cloud) config ---
TRIPO3D_API_KEY = os.getenv("TRIPO3D_API_KEY", "")  # REQUIRED for /image-to-3d/tripo3d
USE_TRIPO_SDK = bool(int(os.getenv("USE_TRIPO_SDK", "1"))) 
TRIPO3D_BASE = os.getenv("TRIPO3D_BASE", "https://api.tripo3d.ai/v2/openapi")
TRIPO3D_MODEL_VERSION = os.getenv("TRIPO3D_MODEL_VERSION", "v2.0-20240919")
TRIPO3D_POLL_SECONDS = float(os.getenv("TRIPO3D_POLL_SECONDS", "2.0"))
TRIPO3D_TIMEOUT_SECONDS = float(os.getenv("TRIPO3D_TIMEOUT_SECONDS", "1800"))  # 30min


app = FastAPI(title="2D→3D Service (Shap-E + Tripo)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/upload", StaticFiles(directory=UPLOAD_DIR), name="upload")
app.mount("/fbupload", StaticFiles(directory=FB_UPLOAD_DIR), name="fbupload")

# ---- Tripo SDK path (robust against schema differences) ----
try:
    from tripo import Client as _TripoClient
    HAVE_TRIPO_SDK = True
except Exception:
    HAVE_TRIPO_SDK = False
    _TripoClient = None


def _sdk_require():
    if not USE_TRIPO_SDK:
        raise RuntimeError("USE_TRIPO_SDK is off")
    if not HAVE_TRIPO_SDK:
        raise RuntimeError("Tripo SDK not installed; add 'tripo' to requirements.txt")
    if not TRIPO3D_API_KEY:
        raise RuntimeError("TRIPO3D_API_KEY not set")


async def _sdk_upload_bytes(data: bytes, filename: str = "image.png"):
    """
    Save bytes to a temp file, upload via SDK, and return the FileToken object.
    """
    _sdk_require()

    def _run():
        import tempfile, os
        suffix = os.path.splitext(filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(data)
            tmp_path = tf.name
        try:
            with _TripoClient(api_key=TRIPO3D_API_KEY) as c:
                tok_obj = c.upload_file(tmp_path)
                return tok_obj
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return await run_in_threadpool(_run)


async def _sdk_create_task(file_token_obj, params: Optional[dict] = None) -> str:
    """
    file_token_obj: the FileToken object returned by SDK upload_file()
    """
    _sdk_require()
    params = params or {}
    mv = os.getenv("TRIPO3D_MODEL_VERSION", "").strip() or "v2.0-20240919"

    def _run():
        with _TripoClient(api_key=TRIPO3D_API_KEY) as c:
            # NOTE: pass as 'file_token=', and pass the FileToken OBJECT
            t = c.image_to_model(file_token=file_token_obj, model_version=mv, **params)
            return getattr(t, "task_id", t)

    return await run_in_threadpool(_run)

# async def _sdk_wait_and_download_glb(task_id: str) -> bytes:
#     """Poll via SDK and return GLB bytes."""
#     _sdk_require()

#     def _run():
#         import time, tempfile, os
#         with _TripoClient(api_key=TRIPO3D_API_KEY) as c:
#             poll_sec = float(os.getenv("TRIPO3D_POLL_SECONDS", "2.0"))
#             while True:
#                 blob = c.try_download_model(task_id)
#                 if blob is None:
#                     time.sleep(poll_sec)

#                 if isinstance(blob, (bytes, bytearray)):
#                     return bytes(blob)
#                 if hasattr(blob, "save"):
#                     tmp_path = None
#                     try:
#                         with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tf:
#                             tmp_path = tf.name
#                         blob.save(tmp_path)
#                         with open(tmp_path, "rb") as f:
#                             return f.read()
#                     finally:
#                         if tmp_path:
#                             try:
#                                 os.remove(tmp_path)
#                             except Exception:
#                                 pass

#                 url = getattr(blob, "url", None)
#                 if url:
#                     r = requests.get(url, timeout=300)
#                     r.raise_for_status()
#                     return r.content

#                 raise RuntimeError(f"Unsupported download blob type: {type(blob)}")

#     return await run_in_threadpool(_run)

async def _sdk_wait_and_download_glb(task_id: str) -> bytes:
    """Poll via SDK and return GLB bytes."""
    _sdk_require()

    def _run():
        import time, tempfile, os
        # respect env-configured timeout / poll interval
        poll_sec = float(os.getenv("TRIPO3D_POLL_SECONDS", "2.0"))
        deadline = time.time() + float(os.getenv("TRIPO3D_TIMEOUT_SECONDS", "1800"))

        with _TripoClient(api_key=TRIPO3D_API_KEY) as c:
            while True:
                blob = c.try_download_model(task_id)

                # Not ready yet -> wait and try again
                if blob is None:
                    if time.time() > deadline:
                        raise TimeoutError("Tripo3D polling timeout")
                    time.sleep(poll_sec)
                    continue  # <<< important

                # Ready -> several possible shapes:
                if isinstance(blob, (bytes, bytearray)):
                    return bytes(blob)

                if hasattr(blob, "save"):  # SDK object with .save(path)
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tf:
                            tmp_path = tf.name
                        blob.save(tmp_path)
                        with open(tmp_path, "rb") as f:
                            return f.read()
                    finally:
                        if tmp_path:
                            try: os.remove(tmp_path)
                            except Exception: pass

                url = getattr(blob, "url", None)
                if url:
                    r = requests.get(url, timeout=300)
                    r.raise_for_status()
                    return r.content

                raise RuntimeError(f"Unsupported download blob type: {type(blob)}")

    return await run_in_threadpool(_run)


# ==========  TripoSR import (local repo) ==========
TRIPOSR_IMPORT_ERROR: Optional[str] = None
HAVE_TRIPOSR = False
_triposr_sys = None 

BASE_DIR = Path(__file__).resolve().parent.parent
TRIPOSR_DIR = BASE_DIR / "external" / "TripoSR"  

os.environ.setdefault("TRIPOSR_MODEL_DIR", str(TRIPOSR_DIR))
# os.environ.setdefault("TRIPOSR_CONFIG", "configs/config.yaml")
# os.environ.setdefault("TRIPOSR_WEIGHTS", "checkpoints/model.ckpt")

try:
    if TRIPOSR_DIR.is_dir() and str(TRIPOSR_DIR) not in sys.path:
        sys.path.insert(0, str(TRIPOSR_DIR))

    spec = importlib.util.find_spec("tsr")
    if spec is None:
        raise ModuleNotFoundError(
            f"'tsr' package not found. Checked sys.path (len={len(sys.path)}). "
            f"Expected at: {TRIPOSR_DIR}"
        )
    from tsr.system import TSR
    HAVE_TRIPOSR = True
except Exception:
    TRIPOSR_IMPORT_ERROR = traceback.format_exc()
    HAVE_TRIPOSR = False
    TSR = None


def _discover_triposr_files(base: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Heuristically find (config_path, weight_path) under the TripoSR repo."""
    candidate_dirs = [
        base,
        base / "configs",
        base / "config",
        base / "assets",
        base / "assets" / "configs",
        base / "checkpoints",
        base / "weights",
        base / "models",
        base / "pretrained",
    ]

    cfg = None
    wgt = None

    for d in candidate_dirs:
        for pat in ("*.yaml", "*.yml", "**/*.yaml", "**/*.yml"):
            matches = list(d.glob(pat))
            if matches:
                cfg = matches[0]
                break
        if cfg:
            break

    for d in candidate_dirs:
        for pat in ("*.ckpt", "*.pth", "*.safetensors", "**/*.ckpt", "**/*.pth", "**/*.safetensors"):
            matches = list(d.glob(pat))
            if matches:
                wgt = matches[0]
                break
        if wgt:
            break

    return cfg, wgt


def get_triposr():
    """
    Lazy-instantiate the TripoSR TSR system once, across multiple fork APIs.
    Supports:
      - TSR()
      - TSR(device=...)
      - TSR.from_pretrained()
      - TSR.from_pretrained(model_dir, config_name, weight_name)
    Environment overrides:
      TRIPOSR_MODEL_DIR, TRIPOSR_CONFIG, TRIPOSR_WEIGHTS
    """
    global _triposr_sys, TRIPOSR_IMPORT_ERROR
    if _triposr_sys is not None:
        return _triposr_sys
    if not HAVE_TRIPOSR or TSR is None:
        return None

    tried = []

    try:
        try:
            _triposr_sys = TSR()
            tried.append("TSR()")
        except TypeError:
            _triposr_sys = TSR(device=DEVICE)
            tried.append("TSR(device=...)")
        except Exception as e_ctor:
            tried.append(f"TSR() failed: {type(e_ctor).__name__}: {e_ctor}")
            _triposr_sys = None

        if _triposr_sys is None and hasattr(TSR, "from_pretrained"):
            sig = inspect.signature(TSR.from_pretrained)
            params = list(sig.parameters.values())
            req_pos = [p for p in params[1:] if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

            if len(req_pos) == 0:
                try:
                    _triposr_sys = TSR.from_pretrained(device=DEVICE)
                    tried.append("TSR.from_pretrained(device=...)")
                except TypeError:
                    _triposr_sys = TSR.from_pretrained()
                    tried.append("TSR.from_pretrained()")
            else:
                model_dir = os.getenv("TRIPOSR_MODEL_DIR")
                config_name = os.getenv("TRIPOSR_CONFIG")
                weight_name = os.getenv("TRIPOSR_WEIGHTS")

                if not (model_dir and config_name and weight_name):
                    cfg_path, wgt_path = _discover_triposr_files(TRIPOSR_DIR)
                    model_dir = model_dir or str(TRIPOSR_DIR)
                    if not config_name and cfg_path:
                        config_name = cfg_path.relative_to(model_dir).as_posix() if cfg_path.is_relative_to(model_dir) else cfg_path.name
                    if not weight_name and wgt_path:
                        weight_name = wgt_path.relative_to(model_dir).as_posix() if wgt_path.is_relative_to(model_dir) else wgt_path.name

                missing = []
                if not model_dir: missing.append("TRIPOSR_MODEL_DIR")
                if not config_name: missing.append("TRIPOSR_CONFIG")
                if not weight_name: missing.append("TRIPOSR_WEIGHTS")
                if missing:
                    raise RuntimeError(
                        "Your TripoSR fork requires explicit pretrained args. "
                        f"Set env vars {missing} or place config/weights under {TRIPOSR_DIR}.\n"
                        f"Tried auto-discovery -> "
                        f"model_dir={model_dir!r}, config_name={config_name!r}, weight_name={weight_name!r}"
                    )

                _triposr_sys = TSR.from_pretrained(model_dir, config_name, weight_name)
                tried.append(f"TSR.from_pretrained({model_dir!r}, {config_name!r}, {weight_name!r})")

        if _triposr_sys is not None and hasattr(_triposr_sys, "to"):
            try:
                _triposr_sys.to(DEVICE)
                tried.append(f".to({DEVICE})")
            except Exception as e_to:
                tried.append(f".to({DEVICE}) failed: {type(e_to).__name__}: {e_to}")

        if _triposr_sys is None:
            raise RuntimeError("Could not instantiate TSR via any known API path.")

        return _triposr_sys

    except Exception:
        TRIPOSR_IMPORT_ERROR = (
            "TripoSR init failed.\n"
            "Tried paths:\n  - " + "\n  - ".join(tried) + "\n\n"
            + traceback.format_exc()
        )
        _triposr_sys = None
        return None


# ==========  Shap-E (diffusers) ==========
from diffusers import ShapEImg2ImgPipeline  # pip install diffusers>=0.28
from diffusers.utils import export_to_ply

shape_pipe: Optional[ShapEImg2ImgPipeline] = None
shape_load_error: Optional[str] = None

def get_shape_pipe() -> Optional[ShapEImg2ImgPipeline]:
    """Lazy-load Shap-E once."""
    global shape_pipe, shape_load_error
    if shape_pipe is not None:
        return shape_pipe
    try:
        model_id = "openai/shap-e-img2img"
        pipe = ShapEImg2ImgPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
        pipe = pipe.to(DEVICE)
        shape_pipe = pipe
        return shape_pipe
    except Exception as e:
        shape_load_error = f"{type(e).__name__}: {e}"
        shape_pipe = None
        return None


# ========== HELPERS ==========
def load_image_from_payload(payload: dict) -> Image.Image:
    """
    Accepts:
      {"url": "<image-url>"} OR {"b64": "<base64-png/jpg>"}.
    Returns a PIL RGB image or raises ValueError.
    """
    if "url" in payload and payload["url"]:
        url = payload["url"]
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            img = Image.open(r.raw).convert("RGB")
            return img
        except Exception as e:
            raise ValueError(f"URL not accessible: {e}")
    elif "b64" in payload and payload["b64"]:
        try:
            raw = base64.b64decode(payload["b64"])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return img
        except Exception:
            raise ValueError("Invalid base64 payload.")
    else:
        raise ValueError("Missing 'url' or 'b64' in payload.")

def mesh_to_glb_bytes(mesh: trimesh.Trimesh) -> bytes:
    """Serialize a trimesh mesh to .glb bytes."""
    return mesh.export(file_type="glb")

def json_error(message: str, stage: str, exc: Optional[BaseException] = None) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": message,
            "stage": stage,
            "trace": traceback.format_exc() if exc else None,
        },
    )

async def _download_bytes(url: str) -> bytes:
    def _get():
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        return r.content
    return await run_in_threadpool(_get)

def _fb_json_url(path: str, params: dict | None = None) -> str:
    if not FIREBASE_DB_URL:
        raise RuntimeError("FIREBASE_DB_URL not set")
    path = path.strip("/")
    q = dict(params or {})
    if FIREBASE_DB_AUTH:
        q["auth"] = FIREBASE_DB_AUTH
    qs = ("?" + urlencode(q, safe=":$,()\"")) if q else ""
    return f"{FIREBASE_DB_URL}/{quote(path)}.json{qs}"

def _fb_fetch(path: str, params: dict | None = None, timeout: int = 20):
    url = _fb_json_url(path, params)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _decode_data_url_to_bytes(data_url: str) -> bytes:
    """
    Accepts 'data:image/...;base64,...' OR raw base64 string.
    Returns decoded bytes (original format).
    """
    s = (data_url or "").strip()
    if s.lower().startswith("data:") and ";base64," in s:
        s = s.split(";base64,", 1)[1]
    return base64.b64decode(s)

def _write_png_to_fb(bytes_: bytes, suggested_name: str | None = None) -> str:
    """
    Convert any image bytes to PNG and save into FB_UPLOAD_DIR.
    File name: <suggested>_<sha16>.png  (dedup by content hash).
    Returns the filename.
    """
    os.makedirs(FB_UPLOAD_DIR, exist_ok=True)
    # load & re-encode to PNG
    img = Image.open(io.BytesIO(bytes_))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    h = hashlib.sha1(bytes_).hexdigest()[:16]
    prefix = (Path(suggested_name).stem + "_") if suggested_name else ""
    fname = f"{prefix}{h}.png"
    fpath = os.path.join(FB_UPLOAD_DIR, fname)
    if not os.path.exists(fpath):
        with open(fpath, "wb") as f:
            img.save(f, format="PNG")
    return fname

async def _tripo3d_upload_from_bytes(data: bytes, filename: str = "image.png", mime: str = "image/png") -> dict:
    """
    Upload to Tripo3D and return dict like:
      {"file_token": "..."}  (preferred for task creation)
      and optionally {"url": "..."} if Tripo returns one
    """
    if not TRIPO3D_API_KEY:
        raise HTTPException(500, "TRIPO3D_API_KEY not set")

    def _upload():
        r = requests.post(
            f"{TRIPO3D_BASE}/upload",
            headers={"Authorization": f"Bearer {TRIPO3D_API_KEY}"},
            files={"file": (filename, data, mime)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    resp = await run_in_threadpool(_upload)
    if resp.get("code") != 0:
        raise HTTPException(502, f"Tripo3D upload error: {resp}")

    d = resp.get("data", {}) or {}

    token = d.get("file_token") or d.get("image_token") or d.get("token")
    url = d.get("url") or d.get("image_url")

    if not token and not url:
        raise HTTPException(502, f"Tripo3D upload returned no file_token or url: {resp}")

    out = {}
    if token:
        out["file_token"] = token
    if url:
        out["url"] = url
    return out


async def _tripo3d_resolve_image_payload(payload: dict) -> dict:
    if payload.get("b64"):
        raw = base64.b64decode(payload["b64"])
        return await _tripo3d_upload_from_bytes(raw)

    if payload.get("url"):
        raw = await _download_bytes(payload["url"])
        mime, fn = "image/png", "image.png"
        low = payload["url"].lower()
        if low.endswith((".jpg", ".jpeg")):
            mime, fn = "image/jpeg", "image.jpg"
        elif low.endswith(".webp"):
            mime, fn = "image/webp", "image.webp"
        return await _tripo3d_upload_from_bytes(raw, filename=fn, mime=mime)

    raise ValueError("Missing 'url' or 'b64' in payload for Tripo3D")

async def tripo3d_create_task(image_spec: dict, params: Optional[dict] = None) -> str:
    """
    image_spec: {"file_token": "..."}  OR  {"url": "..."}  (exactly one)
    Tries multiple payload shapes and with/without model_version.
    Returns task_id on success; otherwise raises HTTPException with the full error matrix.
    """
    if not TRIPO3D_API_KEY:
        raise HTTPException(500, "TRIPO3D_API_KEY not set")

    file_token = image_spec.get("file_token")
    url = image_spec.get("url")
    if not (file_token or url):
        raise HTTPException(400, f"tripo3d_create_task: need file_token or url in {image_spec}")

    mv = os.getenv("TRIPO3D_MODEL_VERSION", "").strip()  # allow empty (=omit)
    user_params = params.copy() if isinstance(params, dict) else {}

    def make_payloads():
        sources = []
        if file_token:
            sources += [
                {"file_token": file_token},
                {"input": {"file_token": file_token}},
            ]
        if url:
            sources += [
                {"url": url},
                {"input": {"url": url}},
            ]

        base_with_mv = ({"type": "image_to_model", "model_version": mv} if mv else None)
        base_without_mv = {"type": "image_to_model"}

        variants = []
        for src in sources:
            if base_with_mv:
                p = {**base_with_mv, **src, **user_params}
                variants.append(("with_mv", p))
            p2 = {**base_without_mv, **src, **user_params}
            variants.append(("no_mv", p2))
        return variants

    attempts = make_payloads()
    tried = []

    for tag, payload in attempts:
        def _post():
            r = requests.post(
                f"{TRIPO3D_BASE}/task",
                headers={
                    "Authorization": f"Bearer {TRIPO3D_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            try:
                data = r.json()
            except Exception:
                data = {"non_json_body": r.text, "status_code": r.status_code}
            return r.status_code, data

        status, data = await run_in_threadpool(_post)

        if status == 200 and isinstance(data, dict) and data.get("code") == 0:
            return data["data"]["task_id"]

        tried.append({"tag": tag, "status": status, "resp": data})

    raise HTTPException(502, f"Tripo3D create task failed with all payload variants: {tried}")


async def tripo3d_poll_until_done(task_id: str) -> dict:
    """
    Poll Tripo3D until completion. Returns the 'output' dict (expects 'model_url').
    """
    start = time.time()
    def _get():
        r = requests.get(
            f"{TRIPO3D_BASE}/task",
            headers={"Authorization": f"Bearer {TRIPO3D_API_KEY}"},
            params={"task_id": task_id},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    while True:
        data = await run_in_threadpool(_get)
        if data.get("code") != 0:
            raise HTTPException(502, f"Tripo3D task error: {data}")
        d = data["data"]
        status = str(d.get("status", "PENDING")).upper()
        if status in {"SUCCESS", "SUCCEEDED", "DONE", "COMPLETED"}:
            return d.get("output") or {}
        if status == "FAILED":
            raise HTTPException(502, f"Tripo3D task failed: {data}")
        if time.time() - start > TRIPO3D_TIMEOUT_SECONDS:
            raise HTTPException(504, "Tripo3D polling timeout")
        await asyncio.sleep(TRIPO3D_POLL_SECONDS)



# ========== HEALTH ==========
@app.get("/health")
def health():
    model_dir = os.getenv("TRIPOSR_MODEL_DIR")
    cfg = os.getenv("TRIPOSR_CONFIG")
    wgt = os.getenv("TRIPOSR_WEIGHTS")
    cfg_exists = bool(model_dir and cfg and os.path.isfile(os.path.join(model_dir, cfg)))
    wgt_exists = bool(model_dir and wgt and os.path.isfile(os.path.join(model_dir, wgt)))
    return {
        "ok": True,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "shape_loaded": shape_pipe is not None,
        "shape_last_error": shape_load_error,
        "triposr_available": HAVE_TRIPOSR,
        "triposr_loaded": _triposr_sys is not None,
        "triposr_import_error": TRIPOSR_IMPORT_ERROR,
        "triposr_model_dir": model_dir,
        "triposr_config": cfg,
        "triposr_weights": wgt,
        "triposr_config_exists": cfg_exists,
        "triposr_weights_exists": wgt_exists,
    }

# ========== UPLOAD ==========
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts multipart form-data 'file' and saves it under UPLOAD_DIR.
    Returns a relative URL you can hand back into /image-to-3d/* endpoints.
    """
    import uuid, shutil
    ext = ""
    if "." in file.filename:
        ext = "." + file.filename.split(".")[-1].lower()
    fname = f"{uuid.uuid4().hex}{ext or '.png'}"
    out_path = os.path.join(UPLOAD_DIR, fname)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": f"/upload/{fname}"}

# ========== UPLOAD GALLERY  ==========
@app.get("/uploads")
def list_uploads(limit: int = 100, offset: int = 0):
    items = []
    for name in os.listdir(UPLOAD_DIR):
        if name.startswith("."):
            continue
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        rel = f"/upload/{name}"
        url = f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel
        items.append({
            "name": name,
            "url": url,
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    # newest first
    items.sort(key=lambda x: x["mtime"], reverse=True)
    total = len(items)
    items = items[offset:offset+limit]
    return {"count": total, "items": items, "limit": limit, "offset": offset}

# ========== FIREBASE -> FBUpload ==========
@app.post("/firebase/sync-to-fbupload")
def firebase_sync_to_fbupload(payload: dict = Body(...)):
    """
    Body:
      { "user": "student_c00", "limit": 200 }
      or
      { "user": "*", "limit": 200 }  # sync all users
    """
    try:
        user = payload.get("user")
        limit = int(payload.get("limit", 200))
        if not FIREBASE_DB_URL:
            return json_error("FIREBASE_DB_URL not configured", stage="firebase-sync-fb")

        def fetch_gallery_for_user(ukey: str):
            path = f"users/{ukey}/gallery"
            data = _fb_fetch(path, params={"orderBy": json.dumps("$key"), "limitToLast": limit})
            return data if isinstance(data, dict) else {}

        users_to_scan = []
        if user == "*" or user == "all":
            # enumerate all users
            users_obj = _fb_fetch("users")
            if isinstance(users_obj, dict):
                users_to_scan = list(users_obj.keys())
        else:
            if not user:
                return json_error("Missing 'user' in payload", stage="firebase-sync-fb")
            users_to_scan = [user]

        public_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
        total_items = []
        for u in users_to_scan:
            data = fetch_gallery_for_user(u)
            ordered = sorted(data.items(), key=lambda kv: kv[0], reverse=True)

            for key, val in ordered:
                if not isinstance(val, dict):
                    continue
                img_data = val.get("image")
                if not img_data:
                    continue

                try:
                    raw = _decode_data_url_to_bytes(img_data)
                    # filename hint: include user and key
                    fname = _write_png_to_fb(raw, suggested_name=f"{u}_{key}")
                except Exception:
                    continue

                rel = f"/fbupload/{fname}"
                url = f"{public_base}{rel}" if public_base else rel
                total_items.append({
                    "user": u,
                    "id": key,
                    "name": fname,
                    "url": url,
                    "prompt": val.get("prompt", ""),
                    "timestamp": val.get("timestamp", ""),
                    "version": val.get("version", ""),
                })

        return {"synced": len(total_items), "items": total_items}
    except Exception as e:
        return json_error("Firebase sync to fbupload failed", stage="firebase-sync-fb", exc=e)

    """
    Body: { "user": "student_c00", "limit": 200 }
    Fetch base64 images from users/<user>/_gallery, convert to PNG,
    save into FB_UPLOAD_DIR, and return their /fbupload URLs.
    """
    try:
        user = payload.get("user")
        limit = int(payload.get("limit", 200))
        if not user:
            return json_error("Missing 'user' in payload", stage="firebase-sync-fb")

        if not FIREBASE_DB_URL:
            return json_error("FIREBASE_DB_URL not configured", stage="firebase-sync-fb")

        path = f"users/{user}/gallery"
        data = _fb_fetch(path, params={"orderBy": json.dumps("$key"), "limitToLast": limit})
        if not isinstance(data, dict):
            return {"synced": 0, "items": []}

        ordered = sorted(data.items(), key=lambda kv: kv[0], reverse=True)
        public_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

        out = []
        for key, val in ordered:
            if not isinstance(val, dict):
                continue
            img_data = val.get("image")
            if not img_data:
                continue

            try:
                raw = _decode_data_url_to_bytes(img_data)
                fname = _write_png_to_fb(raw, suggested_name=key)  # ALWAYS PNG into FB dir
            except Exception:
                continue

            rel = f"/fbupload/{fname}"
            url = f"{public_base}{rel}" if public_base else rel
            out.append({
                "id": key,
                "name": fname,
                "url": url,
                "prompt": val.get("prompt", ""),
                "timestamp": val.get("timestamp", ""),
                "version": val.get("version", ""),
            })

        return {"synced": len(out), "items": out}
    except Exception as e:
        return json_error("Firebase sync to fbupload failed", stage="firebase-sync-fb", exc=e)

@app.get("/fbuploads")
def list_fbuploads(limit: int = 100, offset: int = 0):
    items = []
    for name in os.listdir(FB_UPLOAD_DIR):
        if name.startswith("."):
            continue
        path = os.path.join(FB_UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        st = os.stat(path)
        rel = f"/fbupload/{name}"
        url = f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel
        items.append({
            "name": name,
            "url": url,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    total = len(items)
    items = items[offset:offset+limit]
    return {"count": total, "items": items, "limit": limit, "offset": offset}

# ========== SHAP-E ENDPOINT ==========
@app.post("/image-to-3d/shap-e")
def image_to_3d_shape(
    payload: dict = Body(...),
    guidance_scale: float = 3.0,
    steps: int = 64,
    frame_size: int = 256,
):
    """
    Inputs:
      - JSON {"url": "..."} or {"b64": "..."}
      - Optional query params: guidance_scale, steps, frame_size
    Output:
      - GLB bytes (model/gltf-binary)
    """
    pipe = get_shape_pipe()
    if pipe is None:
        return json_error("Shap-E pipeline not available", stage="load-shape")

    try:
        img = load_image_from_payload(payload)
    except Exception as e:
        return json_error(str(e), stage="image-load", exc=e)

    try:
        result = pipe(
            image=img,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            frame_size=frame_size,
            output_type="mesh",
        )

        with tempfile.TemporaryDirectory() as td:
            ply_path = export_to_ply(result.images[0], os.path.join(td, "model.ply"))
            mesh = trimesh.load(ply_path)  # read as trimesh
            glb_bytes = mesh_to_glb_bytes(mesh)

        headers = {"Content-Disposition": 'attachment; filename="shap-e.glb"'}
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)
    except Exception as e:
        return json_error("Shap-E inference failed", stage="shape-infer", exc=e)


# ========== TRIPOSR ENDPOINT ==========
@app.post("/image-to-3d/triposr")
def image_to_3d_triposr(payload: dict = Body(...), seed: Optional[int] = None):
    """
    Inputs:
      - JSON {"url": "..."} or {"b64": "..."}
      - Optional: seed, mc_resolution, remove_background, foreground_ratio
    Output:
      - GLB bytes (model/gltf-binary)
    """
    tsr = get_triposr()
    if tsr is None:
        msg = "TripoSR not available. Ensure repo is on PYTHONPATH and requirements are installed."
        if TRIPOSR_IMPORT_ERROR:
            msg += f" Import error:\n{TRIPOSR_IMPORT_ERROR}"
        return json_error(msg, stage="load-triposr")

    try:
        img = load_image_from_payload(payload)
    except Exception as e:
        return json_error(str(e), stage="image-load", exc=e)

    try:
        if seed is not None and hasattr(tsr, "set_seed"):
            tsr.set_seed(seed)

        mesh_obj = None

        if callable(tsr) and hasattr(tsr, "extract_mesh"):
            try:
                scene_codes = tsr(image=img, device=DEVICE)
            except TypeError:
                scene_codes = tsr(img, device=DEVICE)
            mc_resolution = 256
            if isinstance(payload, dict) and "mc_resolution" in payload:
                try:
                    mc_resolution = int(payload["mc_resolution"])
                except Exception:
                    pass
            has_vc = True 
            if isinstance(payload, dict) and "has_vertex_color" in payload:
                has_vc = bool(payload["has_vertex_color"])

            try:
                import inspect
                sig = inspect.signature(tsr.extract_mesh)
                params = sig.parameters

                kwargs = {}
                if "resolution" in params:
                    kwargs["resolution"] = mc_resolution

                for flag in ("has_vertex_color", "with_vertex_color", "vertex_color", "with_color"):
                    if flag in params:
                        kwargs[flag] = has_vc
                        break

                meshes = tsr.extract_mesh(scene_codes, **kwargs)
            except TypeError:
                try:
                    meshes = tsr.extract_mesh(scene_codes, mc_resolution, has_vc)
                except Exception as e:
                    raise RuntimeError(f"extract_mesh signature not supported: {e}")

            mesh_obj = meshes[0] if isinstance(meshes, (list, tuple)) and meshes else meshes

        if mesh_obj is None:
            if hasattr(tsr, "reconstruct"):
                mesh_obj = tsr.reconstruct(image=img)
            elif hasattr(tsr, "infer"):
                mesh_obj = tsr.infer(image=img)

        if mesh_obj is None:
            raise RuntimeError(
                "TripoSR API not recognized: no callable+extract_mesh or legacy 'reconstruct'/'infer'."
            )


        if isinstance(mesh_obj, (list, tuple)) and mesh_obj:
            mesh_obj = mesh_obj[0]

        tri: Optional[trimesh.Trimesh] = None
        if isinstance(mesh_obj, trimesh.Trimesh):
            tri = mesh_obj
        elif isinstance(mesh_obj, dict) and "vertices" in mesh_obj and "faces" in mesh_obj:
            tri = trimesh.Trimesh(
                vertices=np.asarray(mesh_obj["vertices"]),
                faces=np.asarray(mesh_obj["faces"]),
                process=False,
            )
        else:
            with tempfile.TemporaryDirectory() as td:
                out_path = os.path.join(td, "model.glb")
                if hasattr(mesh_obj, "export"):
                    try:
                        mesh_obj.export(out_path)
                        tri = trimesh.load(out_path)
                    except Exception:
                        obj_path = os.path.join(td, "model.obj")
                        mesh_obj.export(obj_path)
                        tri = trimesh.load(obj_path)
                elif hasattr(tsr, "export_mesh"):
                    obj_path = os.path.join(td, "model.obj")
                    tsr.export_mesh(mesh_obj, obj_path)
                    tri = trimesh.load(obj_path)
                else:
                    raise RuntimeError("Unsupported TripoSR output; no export() or export_mesh().")

        if tri is None:
            raise RuntimeError("Failed to convert TripoSR output to a mesh.")

        glb_bytes = mesh_to_glb_bytes(tri)
        headers = {"Content-Disposition": 'attachment; filename="triposr.glb"'}
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)

    except Exception as e:
        return json_error("TripoSR inference failed", stage="triposr-infer", exc=e)

# ========== TRIPO3D ENDPOINT (cloud, sync) ==========
@app.post("/image-to-3d/tripo3d")
async def image_to_3d_tripo3d(payload: dict = Body(...)):
    try:
        if USE_TRIPO_SDK:
            if payload.get("b64"):
                raw = base64.b64decode(payload["b64"])
                filename = "image.png"
            elif payload.get("url"):
                raw = await _download_bytes(payload["url"])
                low = payload["url"].lower()
                filename = "image.jpg" if low.endswith((".jpg", ".jpeg")) else "image.webp" if low.endswith(".webp") else "image.png"
            else:
                raise HTTPException(400, "Missing 'url' or 'b64'")

            file_token = await _sdk_upload_bytes(raw, filename=filename)

            params = payload.get("params") if isinstance(payload, dict) else None

            task_id = await _sdk_create_task(file_token, params)
            glb_bytes = await _sdk_wait_and_download_glb(task_id)

            headers = {"Content-Disposition": 'attachment; filename="tripo3d.glb"'}
            return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        return json_error("Tripo3D inference failed", stage="tripo3d", exc=e)
    

# ========== ROOT ==========
@app.get("/")
def root():
    return {
        "service": "2D→3D",
        "endpoints": [
            "/health",
            "/upload", "/uploads",
            "/fbupload", "/fbuploads",
            "/image-to-3d/shap-e",
            "/image-to-3d/triposr",
            "/image-to-3d/tripo3d",
            "/firebase/sync-to-fbupload",
        ],
        "device": DEVICE,
    }


# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
