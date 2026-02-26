# app/services/shape.py
from __future__ import annotations

from typing import Optional

from diffusers import ShapEImg2ImgPipeline
import torch

from app import settings

shape_pipe: Optional[ShapEImg2ImgPipeline] = None
shape_load_error: Optional[str] = None


def get_shape_pipe() -> Optional[ShapEImg2ImgPipeline]:
    """Lazy-load Shap-E once, reuse across requests."""
    global shape_pipe, shape_load_error

    if shape_pipe is not None:
        return shape_pipe

    try:
        model_id = "openai/shap-e-img2img"
        pipe = ShapEImg2ImgPipeline.from_pretrained(model_id, torch_dtype=settings.DTYPE)
        pipe = pipe.to(settings.DEVICE)
        shape_pipe = pipe
        shape_load_error = None
        return shape_pipe
    except Exception as e:
        shape_pipe = None
        shape_load_error = f"{type(e).__name__}: {e}"
        return None