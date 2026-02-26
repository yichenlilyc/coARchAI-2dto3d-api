# app/services/tripo3d.py
from __future__ import annotations

import base64
import time
from typing import Optional, Tuple, Dict, Any

import requests
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from app import settings
from app.services.common import download_bytes, guess_filename_from_url
from app.services.storage import save_generated_glb

# ---- Tripo SDK (optional) ----
try:
    from tripo import Client as _TripoClient
    HAVE_TRIPO_SDK = True
except Exception:
    HAVE_TRIPO_SDK = False
    _TripoClient = None


def _sdk_require():
    if not settings.USE_TRIPO_SDK:
        raise RuntimeError("USE_TRIPO_SDK is off")
    if not HAVE_TRIPO_SDK:
        raise RuntimeError("Tripo SDK not installed; add 'tripo' to requirements.txt")
    if not settings.TRIPO3D_API_KEY:
        raise RuntimeError("TRIPO3D_API_KEY not set")


async def _sdk_upload_bytes(data: bytes, filename: str = "image.png"):
    """
    Save bytes to a temp file, upload via SDK, return FileToken object.
    """
    _sdk_require()

    def _run():
        import tempfile
        import os

        suffix = os.path.splitext(filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(data)
            tmp_path = tf.name

        try:
            with _TripoClient(api_key=settings.TRIPO3D_API_KEY) as c:
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
    Create Tripo task via SDK and return task_id.
    """
    _sdk_require()
    params = params or {}
    mv = (settings.TRIPO3D_MODEL_VERSION or "").strip() or "v2.0-20240919"

    def _run():
        with _TripoClient(api_key=settings.TRIPO3D_API_KEY) as c:
            t = c.image_to_model(file_token=file_token_obj, model_version=mv, **params)
            return getattr(t, "task_id", t)

    return await run_in_threadpool(_run)


async def _sdk_wait_and_download_glb(task_id: str) -> bytes:
    """
    Poll via SDK and return GLB bytes.
    """
    _sdk_require()

    def _run():
        import time
        import tempfile
        import os

        poll_sec = float(settings.TRIPO3D_POLL_SECONDS)
        deadline = time.time() + float(settings.TRIPO3D_TIMEOUT_SECONDS)

        with _TripoClient(api_key=settings.TRIPO3D_API_KEY) as c:
            while True:
                blob = c.try_download_model(task_id)

                if blob is None:
                    if time.time() > deadline:
                        raise TimeoutError("Tripo3D polling timeout")
                    time.sleep(poll_sec)
                    continue

                if isinstance(blob, (bytes, bytearray)):
                    return bytes(blob)

                if hasattr(blob, "save"):
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tf:
                            tmp_path = tf.name
                        blob.save(tmp_path)
                        with open(tmp_path, "rb") as f:
                            return f.read()
                    finally:
                        if tmp_path:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass

                url = getattr(blob, "url", None)
                if url:
                    r = requests.get(url, timeout=300)
                    r.raise_for_status()
                    return r.content

                raise RuntimeError(f"Unsupported download blob type: {type(blob)}")

    return await run_in_threadpool(_run)


# ============================================================
# REST fallback helpers (only used if USE_TRIPO_SDK=0)
# ============================================================

async def _rest_upload_bytes(data: bytes, filename: str, mime: str) -> dict:
    if not settings.TRIPO3D_API_KEY:
        raise HTTPException(500, "TRIPO3D_API_KEY not set")

    def _upload():
        r = requests.post(
            f"{settings.TRIPO3D_BASE}/upload",
            headers={"Authorization": f"Bearer {settings.TRIPO3D_API_KEY}"},
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
        raise HTTPException(502, f"Tripo3D upload returned no token/url: {resp}")

    out = {}
    if token:
        out["file_token"] = token
    if url:
        out["url"] = url
    return out


async def _rest_create_task(image_spec: dict, params: Optional[dict] = None) -> str:
    if not settings.TRIPO3D_API_KEY:
        raise HTTPException(500, "TRIPO3D_API_KEY not set")

    file_token = image_spec.get("file_token")
    url = image_spec.get("url")
    if not (file_token or url):
        raise HTTPException(400, f"Need file_token or url in {image_spec}")

    mv = (settings.TRIPO3D_MODEL_VERSION or "").strip()
    user_params = params.copy() if isinstance(params, dict) else {}

    sources = []
    if file_token:
        sources += [{"file_token": file_token}, {"input": {"file_token": file_token}}]
    if url:
        sources += [{"url": url}, {"input": {"url": url}}]

    attempts = []
    for src in sources:
        if mv:
            attempts.append({"type": "image_to_model", "model_version": mv, **src, **user_params})
        attempts.append({"type": "image_to_model", **src, **user_params})

    tried = []

    for payload in attempts:
        def _post():
            r = requests.post(
                f"{settings.TRIPO3D_BASE}/task",
                headers={
                    "Authorization": f"Bearer {settings.TRIPO3D_API_KEY}",
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

        tried.append({"status": status, "resp": data, "payload": payload})

    raise HTTPException(502, f"Tripo3D create task failed: {tried}")


async def _rest_poll_until_done(task_id: str) -> dict:
    start = time.time()

    def _get():
        r = requests.get(
            f"{settings.TRIPO3D_BASE}/task",
            headers={"Authorization": f"Bearer {settings.TRIPO3D_API_KEY}"},
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

        if time.time() - start > float(settings.TRIPO3D_TIMEOUT_SECONDS):
            raise HTTPException(504, "Tripo3D polling timeout")

        await run_in_threadpool(lambda: time.sleep(float(settings.TRIPO3D_POLL_SECONDS)))


# ============================================================
# Public API used by router
# ============================================================

async def tripo3d_from_payload_sync_glb(payload: dict) -> Tuple[bytes, Dict[str, Any]]:
    """
    Router-facing function:
      - accepts payload {"url": "..."} or {"b64": "..."} and optional "params"
      - returns (glb_bytes, meta)
      - saves to generated/models (sidecar JSON) via save_generated_glb()
    """
    if not isinstance(payload, dict):
        raise HTTPException(400, "Payload must be a JSON object")

    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}

    # Prepare bytes from payload
    if payload.get("b64"):
        raw = base64.b64decode(payload["b64"])
        filename = "image.png"
        mime = "image/png"
        source_url = None
    elif payload.get("url"):
        source_url = payload["url"]
        raw = await download_bytes(source_url)
        filename = guess_filename_from_url(source_url)
        low = source_url.lower()
        if low.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif low.endswith(".webp"):
            mime = "image/webp"
        elif low.endswith(".bmp"):
            mime = "image/bmp"
        else:
            mime = "image/png"
    else:
        raise HTTPException(400, "Missing 'url' or 'b64'")

    # --- SDK path (preferred) ---
    if settings.USE_TRIPO_SDK:
        file_token_obj = await _sdk_upload_bytes(raw, filename=filename)
        task_id = await _sdk_create_task(file_token_obj, params=params)
        glb_bytes = await _sdk_wait_and_download_glb(task_id)
    else:
        # --- REST fallback ---
        image_spec = await _rest_upload_bytes(raw, filename=filename, mime=mime)
        task_id = await _rest_create_task(image_spec, params=params)
        out = await _rest_poll_until_done(task_id)

        model_url = out.get("model_url") or out.get("glb_url") or out.get("url")
        if not model_url:
            raise HTTPException(502, f"Tripo3D output missing model_url: {out}")

        glb_bytes = await download_bytes(model_url)

    meta = save_generated_glb(
        glb_bytes,
        engine="Tripo3D",
        source_url=payload.get("url"),
        seed=None,
        params=params,
    )
    return glb_bytes, meta