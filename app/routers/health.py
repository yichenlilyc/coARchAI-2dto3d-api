from __future__ import annotations

import os

from fastapi import APIRouter

from app import settings
from app.services.shape import get_shape_pipe, shape_load_error
from app.services.triposr import HAVE_TRIPOSR, TRIPOSR_IMPORT_ERROR, _triposr_sys

import torch

router = APIRouter()


@router.get("/health")
def health():
    model_dir = os.getenv("TRIPOSR_MODEL_DIR")
    cfg = os.getenv("TRIPOSR_CONFIG")
    wgt = os.getenv("TRIPOSR_WEIGHTS")
    cfg_exists = bool(model_dir and cfg and os.path.isfile(os.path.join(model_dir, cfg)))
    wgt_exists = bool(model_dir and wgt and os.path.isfile(os.path.join(model_dir, wgt)))

    return {
        "ok": True,
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "shape_loaded": get_shape_pipe() is not None,
        "shape_last_error": shape_load_error,
        "triposr_available": HAVE_TRIPOSR,
        "triposr_loaded": _triposr_sys is not None,
        "triposr_import_error": TRIPOSR_IMPORT_ERROR,
        "triposr_model_dir": model_dir,
        "triposr_config": cfg,
        "triposr_weights": wgt,
        "triposr_config_exists": cfg_exists,
        "triposr_weights_exists": wgt_exists,
        "dictation": {
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "transcribe_model": settings.OPENAI_TRANSCRIBE_MODEL,
            "realtime_model": settings.OPENAI_REALTIME_MODEL,
            "max_upload_mb": settings.DICTATION_MAX_UPLOAD_MB,
            "audio_retention": settings.DICTATION_AUDIO_RETENTION,
            "debug_dir": settings.DICTATION_DEBUG_DIR,
            "debug_dir_exists": os.path.isdir(settings.DICTATION_DEBUG_DIR),
        },
    }