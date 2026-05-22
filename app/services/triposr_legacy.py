# app/services/triposr.py
from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

from app import settings
from app.services.common import mesh_to_glb_bytes


TRIPOSR_IMPORT_ERROR: Optional[str] = None
HAVE_TRIPOSR = False
TSR = None
_triposr_sys = None


def _triposr_repo_dir() -> Path:
    # Requires settings.BASE_DIR (recommended to add in settings.py)
    return Path(settings.BASE_DIR) / "external" / "TripoSR"


def _ensure_triposr_importable():
    global HAVE_TRIPOSR, TSR, TRIPOSR_IMPORT_ERROR

    repo = _triposr_repo_dir()
    os.environ.setdefault("TRIPOSR_MODEL_DIR", str(repo))

    try:
        if repo.is_dir() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        spec = importlib.util.find_spec("tsr")
        if spec is None:
            raise ModuleNotFoundError(
                f"'tsr' package not found. Expected under: {repo} (or installed in site-packages)."
            )

        from tsr.system import TSR as _TSR  # type: ignore
        TSR = _TSR
        HAVE_TRIPOSR = True
    except Exception:
        TRIPOSR_IMPORT_ERROR = traceback.format_exc()
        HAVE_TRIPOSR = False
        TSR = None


def _discover_triposr_files(base: Path) -> tuple[Optional[Path], Optional[Path]]:
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
    Lazy init TSR system, supports multiple fork APIs.
    """
    global _triposr_sys, TRIPOSR_IMPORT_ERROR

    if _triposr_sys is not None:
        return _triposr_sys

    if TRIPOSR_IMPORT_ERROR is None and not HAVE_TRIPOSR:
        _ensure_triposr_importable()

    if not HAVE_TRIPOSR or TSR is None:
        return None

    tried = []
    try:
        try:
            _triposr_sys = TSR()
            tried.append("TSR()")
        except TypeError:
            _triposr_sys = TSR(device=settings.DEVICE)
            tried.append("TSR(device=...)")
        except Exception as e_ctor:
            tried.append(f"TSR() failed: {type(e_ctor).__name__}: {e_ctor}")
            _triposr_sys = None

        if _triposr_sys is None and hasattr(TSR, "from_pretrained"):
            sig = inspect.signature(TSR.from_pretrained)
            params = list(sig.parameters.values())
            req_pos = [
                p for p in params[1:]
                if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]

            if len(req_pos) == 0:
                try:
                    _triposr_sys = TSR.from_pretrained(device=settings.DEVICE)
                    tried.append("TSR.from_pretrained(device=...)")
                except TypeError:
                    _triposr_sys = TSR.from_pretrained()
                    tried.append("TSR.from_pretrained()")
            else:
                repo = _triposr_repo_dir()
                model_dir = os.getenv("TRIPOSR_MODEL_DIR") or str(repo)
                config_name = os.getenv("TRIPOSR_CONFIG")
                weight_name = os.getenv("TRIPOSR_WEIGHTS")

                if not (config_name and weight_name):
                    cfg_path, wgt_path = _discover_triposr_files(repo)
                    if not config_name and cfg_path:
                        try:
                            config_name = cfg_path.relative_to(model_dir).as_posix()
                        except Exception:
                            config_name = cfg_path.name
                    if not weight_name and wgt_path:
                        try:
                            weight_name = wgt_path.relative_to(model_dir).as_posix()
                        except Exception:
                            weight_name = wgt_path.name

                missing = []
                if not config_name:
                    missing.append("TRIPOSR_CONFIG")
                if not weight_name:
                    missing.append("TRIPOSR_WEIGHTS")
                if missing:
                    raise RuntimeError(
                        "TripoSR fork requires explicit pretrained args. "
                        f"Set env vars {missing} or place config/weights under {repo}."
                    )

                _triposr_sys = TSR.from_pretrained(model_dir, config_name, weight_name)
                tried.append(f"TSR.from_pretrained({model_dir!r}, {config_name!r}, {weight_name!r})")

        if _triposr_sys is not None and hasattr(_triposr_sys, "to"):
            try:
                _triposr_sys.to(settings.DEVICE)
                tried.append(f".to({settings.DEVICE})")
            except Exception as e_to:
                tried.append(f".to({settings.DEVICE}) failed: {type(e_to).__name__}: {e_to}")

        if _triposr_sys is None:
            raise RuntimeError("Could not instantiate TSR via any known API path.")

        return _triposr_sys

    except Exception:
        TRIPOSR_IMPORT_ERROR = (
            "TripoSR init failed.\n"
            "Tried paths:\n  - " + "\n  - ".join(tried) + "\n\n" + traceback.format_exc()
        )
        _triposr_sys = None
        return None


def get_triposr_status() -> dict:
    model_dir = os.getenv("TRIPOSR_MODEL_DIR")
    cfg = os.getenv("TRIPOSR_CONFIG")
    wgt = os.getenv("TRIPOSR_WEIGHTS")

    cfg_exists = bool(model_dir and cfg and os.path.isfile(os.path.join(model_dir, cfg)))
    wgt_exists = bool(model_dir and wgt and os.path.isfile(os.path.join(model_dir, wgt)))

    return {
        "triposr_available": HAVE_TRIPOSR,
        "triposr_loaded": _triposr_sys is not None,
        "triposr_import_error": TRIPOSR_IMPORT_ERROR,
        "triposr_model_dir": model_dir,
        "triposr_config": cfg,
        "triposr_weights": wgt,
        "triposr_config_exists": cfg_exists,
        "triposr_weights_exists": wgt_exists,
    }


def triposr_image_to_glb(
    img: Image.Image,
    *,
    seed: Optional[int] = None,
    payload: Optional[dict] = None,
) -> Tuple[bytes, dict]:
    """
    Returns (glb_bytes, debug_info). Router decides saving + response.
    """
    tsr = get_triposr()
    if tsr is None:
        msg = "TripoSR not available."
        if TRIPOSR_IMPORT_ERROR:
            msg += f" Import error:\n{TRIPOSR_IMPORT_ERROR}"
        raise RuntimeError(msg)

    payload = payload or {}

    if seed is not None and hasattr(tsr, "set_seed"):
        tsr.set_seed(seed)

    mesh_obj = None

    # Newer fork style
    if callable(tsr) and hasattr(tsr, "extract_mesh"):
        try:
            scene_codes = tsr(image=img, device=settings.DEVICE)
        except TypeError:
            scene_codes = tsr(img, device=settings.DEVICE)

        mc_resolution = 256
        if "mc_resolution" in payload:
            try:
                mc_resolution = int(payload["mc_resolution"])
            except Exception:
                pass

        has_vc = True
        if "has_vertex_color" in payload:
            has_vc = bool(payload["has_vertex_color"])

        sig = inspect.signature(tsr.extract_mesh)
        params = sig.parameters

        kwargs = {}
        if "resolution" in params:
            kwargs["resolution"] = mc_resolution

        for flag in ("has_vertex_color", "with_vertex_color", "vertex_color", "with_color"):
            if flag in params:
                kwargs[flag] = has_vc
                break

        try:
            meshes = tsr.extract_mesh(scene_codes, **kwargs)
        except TypeError:
            meshes = tsr.extract_mesh(scene_codes, mc_resolution, has_vc)

        mesh_obj = meshes[0] if isinstance(meshes, (list, tuple)) and meshes else meshes

    # Legacy style
    if mesh_obj is None:
        if hasattr(tsr, "reconstruct"):
            mesh_obj = tsr.reconstruct(image=img)
        elif hasattr(tsr, "infer"):
            mesh_obj = tsr.infer(image=img)

    if mesh_obj is None:
        raise RuntimeError("TripoSR API not recognized (no extract_mesh / reconstruct / infer).")

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
        import tempfile
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
        raise RuntimeError("Failed to convert TripoSR output to trimesh.")

    glb_bytes = mesh_to_glb_bytes(tri)

    debug = {
        "engine": "TripoSR",
        "seed": seed,
        "params": {
            "mc_resolution": payload.get("mc_resolution"),
            "has_vertex_color": payload.get("has_vertex_color"),
        },
    }
    return glb_bytes, debug