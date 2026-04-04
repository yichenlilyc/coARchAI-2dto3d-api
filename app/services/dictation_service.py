# app/services/dictation_service.py
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from app import settings


OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"


class DictationServiceError(Exception):
    pass


def _validate_openai_config() -> None:
    if not settings.OPENAI_API_KEY:
        raise DictationServiceError("OPENAI_API_KEY is not configured on the server.")


def _validate_filename(filename: str | None) -> str:
    name = filename or "audio.wav"
    ext = Path(name).suffix.lower()

    if ext and ext not in settings.DICTATION_ALLOWED_AUDIO_EXTS:
        raise DictationServiceError(f"Unsupported audio extension: {ext}")

    return name


def _maybe_save_debug_audio(audio_bytes: bytes, filename: str, when: str) -> None:
    retention = settings.DICTATION_AUDIO_RETENTION.lower()
    if retention == "none":
        return

    if retention == "all" or retention == when:
        safe_ext = Path(filename).suffix.lower() or ".wav"
        debug_name = f"{int(time.time())}_{uuid.uuid4().hex}{safe_ext}"
        debug_path = os.path.join(settings.DICTATION_DEBUG_DIR, debug_name)
        with open(debug_path, "wb") as f:
            f.write(audio_bytes)


async def transcribe_audio_bytes(
    *,
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    language: Optional[str] = None,
    prompt_hint: Optional[str] = None,
) -> dict:
    _validate_openai_config()
    filename = _validate_filename(filename)

    if not audio_bytes:
        raise DictationServiceError("Uploaded audio file is empty.")

    max_bytes = settings.DICTATION_MAX_UPLOAD_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise DictationServiceError(
            f"Audio exceeds max upload size of {settings.DICTATION_MAX_UPLOAD_MB} MB."
        )

    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
    }

    data = {
        "model": settings.OPENAI_TRANSCRIBE_MODEL,
    }

    if language:
        data["language"] = language

    if prompt_hint:
        data["prompt"] = prompt_hint

    files = {
        "file": (filename, audio_bytes, content_type or "application/octet-stream"),
    }

    started = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OPENAI_TRANSCRIPTIONS_URL,
                headers=headers,
                data=data,
                files=files,
            )
    except Exception as e:
        _maybe_save_debug_audio(audio_bytes, filename, "failed_only")
        raise DictationServiceError(f"Failed to reach OpenAI transcription API: {e}") from e

    latency_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        _maybe_save_debug_audio(audio_bytes, filename, "failed_only")
        raise DictationServiceError(
            f"OpenAI transcription failed ({response.status_code}): {response.text}"
        )

    _maybe_save_debug_audio(audio_bytes, filename, "all")

    payload = response.json()
    text = (payload.get("text") or "").strip()

    return {
        "text": text,
        "model": settings.OPENAI_TRANSCRIBE_MODEL,
        "latency_ms": latency_ms,
    }