# Copilot Instructions for coARchAI 2D→3D API

## Architecture Overview

This is a **FastAPI-based containerized service** that wraps three different image-to-3D models:
- **TripoSR** (local GPU/CPU inference) — most flexible, requires model weights
- **Shap-E** (local, via diffusers) — lighter-weight, no external dependencies
- **Tripo3D** (cloud API) — full-featured, requires API key

The server provides HTTP endpoints that accept images (URL or base64) and return GLB 3D models, with metadata saved as JSON sidecars.

### Service Structure
```
scripts/server.py
├─ Shap-E pipeline (lazy-loaded via diffusers)
├─ TripoSR system (lazy-loaded, robust API detection)
├─ Tripo3D cloud client (REST + optional SDK)
├─ FastAPI endpoints (image-to-3d/*)
└─ File storage (upload/, fbupload/, models/ directories)
```

## Critical Developer Workflows

### Running the Service

**Docker (preferred)**
```bash
# GPU: uses NVIDIA CUDA, port 8000
docker compose --profile gpu up --build -d

# CPU: port 8001
docker compose --profile cpu up --build -d

# With Cloudflare Tunnel (public URLs)
docker compose --profile tunnel up -d
```

**Local development (Python 3.11+)**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
```

### Health Checks
```bash
# GPU: http://localhost:8000/health
# CPU: http://localhost:8001/health
```
Health response includes model availability, device status, and TripoSR file existence checks.

### Key Environment Variables
- **TRIPOSR_MODEL_DIR**, **TRIPOSR_CONFIG**, **TRIPOSR_WEIGHTS** — required for TripoSR inference
- **TRIPO3D_API_KEY**, **USE_TRIPO_SDK** — required for `/image-to-3d/tripo3d` endpoint
- **FIREBASE_DB_URL**, **FIREBASE_DB_AUTH** — optional, for Firebase RTDB sync
- **PUBLIC_BASE_URL** — affects returned model URLs (e.g., "http://localhost:8000")

## Project-Specific Patterns & Conventions

### 1. **Lazy Loading of Heavy Models**
Models are loaded on-first-use, not at startup. This prevents startup delays and memory waste if a model isn't used.

**Pattern (seen in `get_shape_pipe()`, `get_triposr()`):**
- Global `_cache_var` tracks loaded state
- Check cache, return early if loaded
- On load failure, cache the error message (not exception) for health endpoint
- Use `try/except` with detailed error tracing for diagnostics

**Example from TripoSR loading:**
```python
def get_triposr():
    global _triposr_sys, TRIPOSR_IMPORT_ERROR
    if _triposr_sys is not None:
        return _triposr_sys
    # ... attempt multiple API signatures ...
    if _triposr_sys is None:
        TRIPOSR_IMPORT_ERROR = traceback.format_exc()
    return _triposr_sys
```

### 2. **Multi-Signature Function Support**
TripoSR and similar models may have varying constructor/method signatures across versions. The code handles this gracefully by:
- Attempting multiple API shapes (e.g., `TSR()`, `TSR(device=...)`, `TSR.from_pretrained(...)`)
- Using `inspect.signature()` to detect available parameters
- Maintaining a `tried` list for detailed error reporting

**When adding model support:** Test against multiple versions of the library; document which signatures work.

### 3. **Error Responses as JSON with Context**
All inference errors return a consistent JSON structure:
```python
{
    "error": "<user-friendly message>",
    "stage": "<phase name: image-load, shape-infer, triposr-infer, etc>",
    "trace": "<full traceback if exception provided>"
}
```
See `json_error()` helper. This allows clients to debug which step failed.

### 4. **Model Output Persistence**
Every successful inference saves:
- `.glb` file (the 3D mesh)
- `.json` sidecar with metadata (engine, source URL, seed, params, size, mtime)

Pattern:
```python
meta = _save_model_glb(glb_bytes, engine="ShapE", source_url=url, seed=seed, params={...})
# Returns: {"name": "20251031T...glb", "url": "...", "engine": "ShapE", ...}
```

Metadata is used by `/generated/models` list endpoint and archive/download features.

### 5. **Image Input Flexibility**
All image-to-3d endpoints accept either:
- `{"url": "http://..."}` — fetched via `requests.get()`
- `{"b64": "base64-encoded-bytes"}` — decoded via `base64.b64decode()`

Helper `load_image_from_payload()` unifies this; converts to PIL RGB image.

### 6. **Firebase RTDB Integration**
The `/firebase/sync-to-fbupload` endpoint pulls gallery images from Firebase RTDB, converts to PNG, and saves locally for serving.

**Key differences from standard RTDB calls:**
- Uses REST API with `.json` suffix: `{FIREBASE_DB_URL}/{path}.json?auth=...`
- Handles both single-user and all-users sync (`user="*"`)
- Images may be data URLs (`data:image/...;base64,...`) or raw base64; helper `_decode_data_url_to_bytes()` handles both

### 7. **Tripo3D Cloud Workflow**
Unlike local models, Tripo3D is async task-based:
1. Upload image → get file token
2. Create task (with optional params) → get task_id
3. Poll until completion → download GLB

Functions provided:
- `_sdk_upload_bytes()` — uses Tripo SDK if available
- `_tripo3d_upload_from_bytes()` — uses REST API as fallback
- `tripo3d_create_task()` — tries multiple payload variants (robustness against API changes)
- `tripo3d_poll_until_done()` — respects env-configured timeout/poll intervals

**Robustness pattern:** When Tripo3D API shape is uncertain, we generate multiple `(tag, payload)` variants and try each until one succeeds (see `make_payloads()` in `tripo3d_create_task()`).

### 8. **Static File Serving & Cleanup**
Three directories are served and listed:
- `/upload/` — user-uploaded images for reconstruction
- `/fbupload/` — Firebase-synced images (PNG-normalized)
- `/model/` — saved GLB + JSON sidecars

Delete endpoints use `_safe_join()` to prevent path traversal attacks:
```python
p = _safe_join(MODELS_DIR, filename)  # ensures filename is plain basename
```

## Integration Points & External Dependencies

### PyTorch + CUDA
- Device detection: `torch.cuda.is_available()` and `DTYPE` (float16 if CUDA, else float32)
- Loaded at import; absence won't crash server but health check reflects status

### Diffusers (Shap-E)
- Model: `openai/shap-e-img2img` (auto-downloaded on first use)
- Output: PLY images that are converted to trimesh, then GLB

### TripoSR (External Repo)
- **Location:** `external/TripoSR/` (Git submodule or manual clone)
- **Key pattern:** Dynamically adds to `sys.path` and imports `tsr` module
- **Model files:** Must be placed at `external/TripoSR/{config.yaml, model.ckpt}`
- If TripoSR is not available, endpoints gracefully fail with diagnostic message

### Tripo3D
- **SDK:** `tripo` package (optional; `USE_TRIPO_SDK=1` enables it)
- **REST fallback:** Always available if API key set; SDK just simplifies polling
- **Auth:** Bearer token in `Authorization` header

### CORS & Static Files
- All origins allowed (CORS permissive)
- Files mounted at `/upload`, `/fbupload`, `/model` (read-only serving)

## Code Organization Tips

### Adding a New Image-to-3D Model
1. **Lazy-load function:** Create `get_new_model()` following the pattern (cache + error tracking)
2. **Endpoint:** Add `@app.post("/image-to-3d/<name>")` that calls `load_image_from_payload()` → inference → `_save_model_glb()`
3. **Health check:** Add model status to `/health` response
4. **Error handling:** Use `json_error(msg, stage="...", exc=e)` for consistent error reporting

### Modifying Inference Parameters
- Query params go into endpoint signature: `@app.post("/image-to-3d/shap-e", guidance_scale: float = 3.0)`
- Payload params extracted from body: `payload.get("mc_resolution")`
- Always pass both to `_save_model_glb()` so metadata is preserved

### Handling New Tripo3D API Changes
- Update `make_payloads()` in `tripo3d_create_task()` to include new variants
- Payload variants are tried sequentially until one succeeds (robust against version mismatches)
- Add new env vars to docker-compose.yml if needed (e.g., new model version)

## Testing & Debugging

### Health Endpoint Diagnostics
```bash
curl http://localhost:8000/health | jq .
```
Shows which models are loaded, any import errors, model file paths, and CUDA status.

### Test Endpoints
```bash
# Upload test image
curl -F "file=@test.png" http://localhost:8000/generated/images

# Shap-E (light inference, no GPU usually needed)
curl -X POST http://localhost:8000/image-to-3d/shap-e \
  -H "Content-Type: application/json" \
  -d '{"url":"http://localhost:8000/upload/...png"}' \
  -o out.glb

# TripoSR (requires model weights)
curl -X POST http://localhost:8000/image-to-3d/triposr \
  -H "Content-Type: application/json" \
  -d '{"url":"http://localhost:8000/upload/...png"}' \
  -o out.glb
```

### Docker Logs
```bash
# GPU container
docker compose logs -f app-gpu

# See uvicorn startup and request logs
```

### Common Issues & Solutions
- **TripoSR import fails:** Verify `external/TripoSR/` exists and `tsr` package is in requirements
- **CUDA out of memory:** Reduce batch size or use CPU container; check GPU memory: `nvidia-smi`
- **Tripo3D task timeout:** Increase `TRIPO3D_TIMEOUT_SECONDS` in `.env`
- **Models not found in archive download:** Check `/generated/models` endpoint response; ensure GLB + JSON sidecars exist

## File Reference

| File | Purpose |
|------|---------|
| `scripts/server.py` | Main FastAPI application (1400+ lines) |
| `docker-compose.yml` | GPU/CPU service definitions; tunnel config |
| `requirements.txt` | Python dependencies (diffusers, fastapi, trimesh, etc.) |
| `requirements.cpu.txt` | CPU-only variant (no CUDA torch) |
| `external/TripoSR/` | Git submodule or manual clone of TripoSR repo |
| `cloudflared/config.yml` | Cloudflare Tunnel ingress routes |
| `README.md` | User-facing setup & API documentation |

---

**Last updated:** Feb 2026 | **Target audience:** AI agents, contributors, and maintainers
