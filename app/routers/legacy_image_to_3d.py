# app/routers/legacy_image_to_3d.py
from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from app.services.common import load_image_from_payload
from app.services.storage import save_generated_glb

from app.services.shape import get_shape_pipe, shape_load_error
from app.services.triposr import get_triposr, TRIPOSR_IMPORT_ERROR
from app.services.tripo3d import tripo3d_from_payload_sync_glb

from app.services.errors import json_error

router = APIRouter()


@router.post("/image-to-3d/shap-e")
def image_to_3d_shape(
    payload: dict = Body(...),
    guidance_scale: float = 3.0,
    steps: int = 64,
    frame_size: int = 256,
):
    pipe = get_shape_pipe()
    if pipe is None:
        return json_error(f"Shap-E pipeline not available: {shape_load_error}", stage="load-shape")

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

        from diffusers.utils import export_to_ply
        import trimesh

        with tempfile.TemporaryDirectory() as td:
            ply_path = export_to_ply(result.images[0], os.path.join(td, "model.ply"))
            mesh = trimesh.load(ply_path)
            glb_bytes = mesh.export(file_type="glb")

        meta = save_generated_glb(
            glb_bytes,
            engine="ShapE",
            source_url=payload.get("url"),
            seed=None,
            params={"guidance_scale": guidance_scale, "steps": steps, "frame_size": frame_size},
        )
        headers = {
            "Content-Disposition": 'attachment; filename="shap-e.glb"',
            "X-Model-URL": meta["url"],
        }
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)
    except Exception as e:
        return json_error("Shap-E inference failed", stage="shape-infer", exc=e)


@router.post("/image-to-3d/triposr")
def image_to_3d_triposr(payload: dict = Body(...), seed: Optional[int] = None):
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
        from app.services.triposr import triposr_image_to_glb_bytes
        glb_bytes, used_params = triposr_image_to_glb_bytes(tsr, img, payload=payload, seed=seed)

        meta = save_generated_glb(
            glb_bytes,
            engine="TripoSR",
            source_url=payload.get("url"),
            seed=seed,
            params=used_params,
        )
        headers = {
            "Content-Disposition": 'attachment; filename="triposr.glb"',
            "X-Model-URL": meta["url"],
        }
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)
    except Exception as e:
        return json_error("TripoSR inference failed", stage="triposr-infer", exc=e)


@router.post("/image-to-3d/tripo3d")
async def image_to_3d_tripo3d(payload: dict = Body(...)):
    try:
        glb_bytes, meta = await tripo3d_from_payload_sync_glb(payload)
        headers = {
            "Content-Disposition": 'attachment; filename="tripo3d.glb"',
            "X-Model-URL": meta["url"],
        }
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        return json_error("Tripo3D inference failed", stage="tripo3d", exc=e)