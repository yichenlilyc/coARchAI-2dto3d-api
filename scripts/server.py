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

from fastapi import FastAPI, Body
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIG / DEVICE ---
USE_CUDA = bool(int(os.getenv("USE_CUDA", "0")))  # default 0 (CPU); set 1 to prefer CUDA if available
CUDA_OK = torch.cuda.is_available() and USE_CUDA
DEVICE = "cuda" if CUDA_OK else "cpu"
DTYPE = torch.float16 if CUDA_OK else torch.float32

app = FastAPI(title="2D→3D Service (Shap-E + TripoSR)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ========== ROOT ==========
@app.get("/")
def root():
    return {
        "service": "2D→3D",
        "endpoints": ["/health", "/image-to-3d/shap-e", "/image-to-3d/triposr"],
        "device": DEVICE,
    }


# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
