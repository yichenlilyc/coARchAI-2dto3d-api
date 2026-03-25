# app/routers/dictation.py
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.dictation_service import (
    DictationServiceError,
    transcribe_audio_bytes,
)

router = APIRouter(prefix="/dictation")
log = logging.getLogger(__name__)


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    section_index: Optional[int] = Form(None),
    language: Optional[str] = Form(None),
    prompt_hint: Optional[str] = Form(None),
    provider_variant: Optional[str] = Form(None),
):
    try:
        audio_bytes = await file.read()

        result = await transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            content_type=file.content_type,
            language=language,
            prompt_hint=prompt_hint,
        )

        log.info(
            "dictation_transcribe_ok session=%s section=%s variant=%s model=%s bytes=%s latency_ms=%s",
            session_id,
            section_index,
            provider_variant,
            result["model"],
            len(audio_bytes),
            result["latency_ms"],
        )

        return {
            "ok": True,
            "text": result["text"],
            "provider": "openai",
            "model": result["model"],
            "session_id": session_id,
            "section_index": section_index,
            "latency_ms": result["latency_ms"],
        }

    except DictationServiceError as e:
        log.exception(
            "dictation_transcribe_failed session=%s section=%s",
            session_id,
            section_index,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception(
            "dictation_transcribe_unhandled session=%s section=%s",
            session_id,
            section_index,
        )
        raise HTTPException(status_code=500, detail="Internal transcription error.")