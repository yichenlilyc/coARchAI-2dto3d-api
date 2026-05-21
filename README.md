# CoARchAI 2D-to-3D API — RunPod Deployment Guide

Containerized **FastAPI** server powering image-to-3D generation using **Meta SAM 3**, **SAM 3D Objects**, and **Tripo3D**. Designed for deployment on RunPod with a Cloudflare Tunnel for public access.

The stack runs two services in a single container:
- **Main API** (port 8000, GPU 0) — SAM 3 segmentation, Tripo3D cloud generation, uploads, presets, dictation
- **SAM 3D Worker** (port 8001, GPU 1) — internal worker that runs the heavy SAM 3D Objects mesh generation pipeline

---

## Contents

- [Hardware Requirements](#hardware-requirements)
- [Building the Docker Image](#building-the-docker-image)
- [Hugging Face Token & Gated Models](#hugging-face-token--gated-models)
- [API Overview](#api-overview)
- [Cloudflare Tunnel Setup](#cloudflare-tunnel-setup)
- [RunPod Template Setup](#runpod-template-setup)
- [Firebase Project Setup](#firebase-project-setup)
- [Firebase Storage & Metadata Indexing](#firebase-storage--metadata-indexing)
- [Environment Variable Reference](#environment-variable-reference)
- [GPU Architecture Reference](#gpu-architecture-reference)

---

## Hardware Requirements

This deployment requires a **dual-GPU RunPod instance**. The launcher explicitly splits work across two GPUs:

| Process | GPU | Port | Minimum VRAM |
|---|---|---|---|
| Main API (SAM 3 segmentation) | GPU 0 | 8000 | 16 GB |
| SAM 3D Worker (mesh generation) | GPU 1 | 8001 | 16 GB |

**Total minimum: 32 GB VRAM across two GPUs.**

A single high-VRAM GPU (e.g. A100 80 GB) can be used by adjusting `CUDA_VISIBLE_DEVICES` in `launcher.py`, but the default configuration expects two separate devices.

### Supported GPU Architectures

The Dockerfile pre-compiles CUDA kernels for the following compute capabilities:

| GPU Family | Compute Capability | Example GPUs |
|---|---|---|
| Turing | 7.5 | RTX 2080 Ti, T4 |
| Ampere (data center) | 8.0 | A100, A30 |
| Ampere (consumer) | 8.6 | RTX 3090, 3080, A6000 |
| Ada Lovelace | 8.9 | RTX 4090, 4080, L40S |
| Hopper | 9.0 | H100 |

Recommended RunPod GPU types: **2x A100 (40/80 GB)**, **2x RTX 4090**, or **1x H100 (80 GB SXM)**.

Minimum CUDA version required: **CUDA 12.6+** (the image is built on CUDA 12.8).

---

## Building the Docker Image

The active Dockerfile for RunPod deployment is [`docker/Dockerfile.sam3.runpod`](docker/Dockerfile.sam3.runpod).

### Prerequisites

- Docker with BuildKit enabled
- A valid Hugging Face token (see [next section](#hugging-face-token--gated-models))

### Build Command

The Hugging Face token must be passed as a build argument. The image downloads model weights during the build so they are baked in and ready at container start.

```bash
docker build \
  --build-arg HF_TOKEN=hf_your_token_here \
  -f docker/Dockerfile.sam3.runpod \
  -t coarchai-2dto3d:latest \
  .
```

### What the Build Does

1. **Base image** — `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
2. **Installs Miniforge** (conda) and sets up Python 3.12
3. **Installs PyTorch 2.7.0** with CUDA 12.6 wheels
4. **Clones and installs SAM 3** from `facebookresearch/sam3`
5. **Clones SAM 3D Objects** and builds it in a dedicated `sam3d-objects` conda environment with CUDA kernel compilation
6. **Downloads model weights** from Hugging Face using `scripts/download_models.py` (requires `HF_TOKEN`)
7. **Installs cloudflared** for the Cloudflare Tunnel
8. **Copies the application** and sets all runtime environment variables

### Pushing to a Registry (for RunPod)

```bash
docker tag coarchai-2dto3d:latest your-registry/coarchai-2dto3d:latest
docker push your-registry/coarchai-2dto3d:latest
```

RunPod supports Docker Hub and any public/private registry. Use the image URL when creating your pod template.

---

## Hugging Face Token & Gated Models

**A Hugging Face account with access to the following gated repositories is required.** Request access on the Hugging Face model pages before building — the build will fail at the download step otherwise.

| Model | Repository | Purpose |
|---|---|---|
| SAM 3 | `facebookresearch/sam3` (GitHub, cloned at build) | 2D segmentation via text prompt and touch/click |
| SAM 2.1 Large | `facebook/sam2.1-hiera-large` | Segmentation weights used by the SAM 3 pipeline |
| SAM 3D Objects | `facebook/sam-3d-objects` | Full 3D mesh generation from masked images |

All three models are gated (require Meta's approval on Hugging Face). After approval is granted:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with **Read** access
3. Pass it as `--build-arg HF_TOKEN=hf_...` when building the image

The token is only needed at **build time** for the model download step. It does not need to be set as a runtime environment variable on the RunPod pod.

---

## API Overview

The API is built with FastAPI and served by Uvicorn on port 8000. All endpoints are accessible at your public tunnel URL or `http://localhost:8000` locally.

Interactive docs are available at `/docs` once the container is running.

### Health Check

```
GET /health
```

Returns the status of all loaded models, CUDA availability, and service configuration. Use this to verify the container started correctly.

### SAM 3 Segmentation

**Touch/Click to Mask**
```
POST /image-to-3d/sam/segment-touch
```
Body: `{ "b64": "<base64-image>", "x": <pixel-x>, "y": <pixel-y> }`
Returns: `{ "mask_b64": "<base64-png-mask>" }`

**Text Prompt to Mask**
```
POST /image-to-3d/segment-prompt
```
Body: `{ "b64": "<base64-image>", "prompt": "chair" }`
Returns: `{ "mask_b64": "<base64-png-mask>" }`

### SAM 3D Mesh Generation (Async)

**Submit a generation job**
```
POST /image-to-3d/sam/generate
```
Body: `{ "b64": "<base64-image>", "mask_b64": "<base64-mask>" }`
Returns: `{ "job_id": "<uuid>", "status": "queued" }`

The job is handed off to the SAM 3D Worker on GPU 1 and processed asynchronously.

**Poll for status**
```
GET /image-to-3d/sam/status/{job_id}?format=glb&user_id=<uid>
```
Returns `{ "status": "running" }` while processing, or `{ "status": "succeeded", "model_url": "<firebase-url>" }` when complete. Supported formats: `glb`, `ply`.

### Uploads & Presets

| Endpoint | Description |
|---|---|
| `POST /uploads/...` | Upload images for processing |
| `GET /presets/maps/...` | Serve preset map files |
| `GET /presets/models/...` | Serve preset 3D models |
| `GET /static/generated/...` | Serve generated outputs |

### Dictation

```
POST /dictation/...
```

Transcribes audio via OpenAI Whisper. Requires `OPENAI_API_KEY` to be set.

---

## Cloudflare Tunnel Setup

The container includes `cloudflared` and will automatically start a tunnel at launch if `CF_TUNNEL_TOKEN` is set in the pod's environment.

### Creating a Tunnel

1. Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/) → **Networks** → **Tunnels**
2. Click **Create a tunnel** → choose **Cloudflared**
3. Name the tunnel (e.g. `coarchai-api`) and save
4. Copy the tunnel token — a long string starting with `eyJ...`
5. Under the tunnel's **Public Hostname** tab, add a route:
   - **Subdomain**: your chosen subdomain (e.g. `api`)
   - **Domain**: your Cloudflare-managed domain
   - **Service**: `HTTP` → `localhost:8000`

### Connecting the Tunnel at Runtime

Set `CF_TUNNEL_TOKEN` as a RunPod environment variable (see [RunPod Template Setup](#runpod-template-setup)). The `launcher.py` startup script detects the token and starts the tunnel automatically:

```python
# From launcher.py
cf_token = os.environ.get("CF_TUNNEL_TOKEN")
if cf_token:
    cf_process = subprocess.Popen([
        "cloudflared", "tunnel", "--protocol", "http2",
        "--no-autoupdate", "run", "--token", cf_token
    ])
```

Also set `PUBLIC_BASE_URL` to your tunnel's public hostname so the API generates correct absolute URLs in responses and the `/docs` interface uses the right server:

```
PUBLIC_BASE_URL=https://api.yourdomain.com
```

### Verifying the Tunnel

After the pod starts, check the health endpoint through the tunnel:

```bash
curl https://api.yourdomain.com/health
```

---

## RunPod Template Setup

### Creating a Custom Template

1. In RunPod, go to **My Templates** → **New Template**
2. Set **Container Image** to your pushed image (e.g. `docker.io/youruser/coarchai-2dto3d:latest`)
3. Set **Container Disk** to at least **50 GB** (model weights alone are ~20 GB)
4. Set **Expose HTTP Ports** to `8000`
5. Add all required environment variables in the **Environment Variables** section (see below)

### Selecting the Right Pod

When deploying from the template, filter pods by:
- **GPU VRAM**: 32 GB minimum total (ideally 2x 16 GB+ or a single large GPU)
- **GPU Count**: 2 (the launcher assigns GPU 0 to the API and GPU 1 to the worker by default)
- **CUDA Version**: 12.6 or higher

Recommended instance types: `2x A100 SXM (40 GB)`, `2x RTX 4090`, `1x H100 (80 GB SXM)`.

---

## Firebase Project Setup

This section covers creating the Firebase project, enabling the required services, and generating the credentials the API expects.

### 1. Create a Firebase Project

1. Go to [console.firebase.google.com](https://console.firebase.google.com) and click **Add project**
2. Give it a name (e.g. `coarchai`), disable Google Analytics if not needed, and click **Create project**

### 2. Enable Cloud Firestore

1. In the left sidebar, go to **Build** → **Firestore Database**
2. Click **Create database**
3. Choose **Production mode** (you will set rules in a moment)
4. Select a region close to your RunPod datacenter and click **Enable**

Once created, set the following security rules under **Firestore Database** → **Rules** to allow the server-side service account full access while keeping unauthenticated clients read-only:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /models_3d/{modelId} {
      allow read: if true;
      allow write: if request.auth != null;
    }
  }
}
```

The API writes to Firestore using the Admin SDK (bypasses rules entirely), so these rules only affect client-side access from your mobile/web app.

**Firestore Index**

If your client app queries models by user — e.g. `where("user_id", "==", uid).orderBy("created_at", "desc")` — you will need a composite index. Create it in **Firestore Database** → **Indexes** → **Composite** → **Add index**:

| Collection | Field | Order |
|---|---|---|
| `models_3d` | `user_id` | Ascending |
| `models_3d` | `created_at` | Descending |

Firestore will also prompt you to create missing indexes automatically the first time a query fails — follow the link in the error message.

### 3. Enable Firebase Storage

1. In the left sidebar, go to **Build** → **Storage**
2. Click **Get started**, choose **Production mode**, select a region, and click **Done**

The API calls `blob.make_public()` after each upload, which grants public read access to each file individually. For this to work, the Storage bucket's IAM must allow the `allUsers` principal to have the **Storage Object Viewer** role. Firebase does not set this by default.

To enable it:

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → **Cloud Storage** → **Buckets**
2. Click your bucket (it will be named something like `your-project.firebasestorage.app`)
3. Go to the **Permissions** tab → **Grant access**
4. New principals: `allUsers`
5. Role: `Storage Object Viewer`
6. Click **Save** (dismiss the public access warning)

Alternatively, set the Storage rules to allow public reads in the Firebase console under **Storage** → **Rules**:

```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /models/{userId}/{modelId} {
      allow read: if true;
      allow write: if request.auth != null;
    }
  }
}
```

### 4. Create a Service Account

The API authenticates with Firebase using a service account, not a user login. The service account credentials are what you encode into `FIREBASE_SERVICE_ACCOUNT_B64`.

1. In the Firebase console, go to **Project Settings** (gear icon) → **Service accounts**
2. Click **Generate new private key** → **Generate key**
3. A `.json` file will download — keep it secure and do not commit it to git

The downloaded JSON looks like:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
  "client_email": "firebase-adminsdk-...@your-project.iam.gserviceaccount.com",
  ...
}
```

### 5. Encode the Credentials for RunPod

Base64-encode the JSON file into a single-line string:

```bash
base64 -i your-service-account.json | tr -d '\n'
```

Copy the output. This is the value you paste into `FIREBASE_SERVICE_ACCOUNT_B64` in your RunPod template environment variables. The API decodes it at startup:

```python
cert_dict = json.loads(base64.b64decode(firebase_b64).decode('utf-8'))
cred = credentials.Certificate(cert_dict)
initialize_app(cred, {'storageBucket': 'your-project.firebasestorage.app'})
```

### 6. Update the Bucket Name in `database.py`

The storage bucket name is set directly in [`app/database.py`](app/database.py). Update it to match your Firebase project's bucket:

```python
initialize_app(cred, {
    'storageBucket': 'your-project-id.firebasestorage.app'
})
```

The bucket name follows the pattern `<project-id>.firebasestorage.app` and is visible in the Firebase console under **Storage** → **Files** in the top bar.

---

## Firebase Storage & Metadata Indexing

The API uses two Firebase products for model persistence:

- **Firebase Storage** — stores the generated `.glb` and `.ply` mesh files at a public URL
- **Cloud Firestore** — indexes metadata for every generated model so clients can query by user or model ID

### Authentication

Firebase is initialized using a **service account JSON** encoded as a base64 string. This avoids having to mount a credentials file into the container and lets you paste it directly as a RunPod environment variable.

To generate the value from your service account file:

```bash
base64 -i your-firebase-service-account.json | tr -d '\n'
```

Paste the output as the value of `FIREBASE_SERVICE_ACCOUNT_B64` in your RunPod template. Without this variable, Firebase will not initialize and all calls to `save_model_to_firebase` will raise an exception.

### Storage Layout

Generated mesh files are uploaded to Firebase Storage under the path:

```
models/{user_id}/{model_id}.{format}
```

Where:
- `user_id` is passed by the client when polling the status endpoint (`?user_id=...`)
- `model_id` is the task UUID assigned at generation time
- `format` is `glb` or `ply`

Both formats are uploaded and made public independently. The public URL returned by Storage is then written into Firestore.

### Firestore Metadata Document

After each upload, a document is written to the `models_3d` Firestore collection using `model_id` as the document ID. The write uses `merge=True`, so calling the status endpoint for both `glb` and `ply` formats merges both URLs into the same document without overwriting:

```json
{
  "model_id": "<task-uuid>",
  "user_id": "<user-id>",
  "source_image_id": "<original-image-url-or-id>",
  "preview_url": "<optional-preview-image-url>",
  "glb_url": "https://storage.googleapis.com/...",
  "ply_url": "https://storage.googleapis.com/...",
  "created_at": "2024-01-15T10:30:00.000Z",
  "status": "completed"
}
```

This document structure allows clients to:
- Look up a model by `model_id` directly
- Query all models for a `user_id` via a Firestore `where` filter
- Retrieve both mesh formats from a single document

### Legacy Realtime Database

The `FIREBASE_DB_URL` and `FIREBASE_DB_AUTH` variables are used by the legacy router for reading data from Firebase Realtime Database via the REST API. This is separate from the Firestore + Storage pipeline above and is only needed if the legacy image endpoints are in use.

---

## Environment Variable Reference

Set these in the RunPod template's **Environment Variables** section. Variables marked **Required** will cause the service to fail or degrade without them.

### Tunnel & Public Access

| Variable | Required | Description |
|---|---|---|
| `CF_TUNNEL_TOKEN` | Required | Cloudflare Tunnel token from the Zero Trust dashboard. Without this, the container runs with no public URL. |
| `PUBLIC_BASE_URL` | Recommended | Full public URL (e.g. `https://api.yourdomain.com`). Used to generate absolute URLs in API responses and configure the FastAPI `/docs` server list. |

### AI Services

| Variable | Required | Description |
|---|---|---|
| `TRIPO3D_API_KEY` | For Tripo3D | API key from [tripo3d.ai](https://www.tripo3d.ai). Required to use the Tripo3D cloud generation endpoint. |
| `OPENAI_API_KEY` | For Dictation | OpenAI API key. Required to use the audio transcription endpoint. |

### Firebase (for model storage & retrieval)

| Variable | Required | Description |
|---|---|---|
| `FIREBASE_SERVICE_ACCOUNT_B64` | Required for Firebase | Base64-encoded Firebase service account JSON. Initializes both Firestore and Firebase Storage. Generate with `base64 -i service-account.json \| tr -d '\n'`. |
| `FIREBASE_DB_URL` | For legacy endpoints | Firebase Realtime Database URL (e.g. `https://your-project.firebaseio.com`). Only needed for the legacy image router. |
| `FIREBASE_DB_AUTH` | For legacy endpoints | Firebase RTDB authentication secret. Only needed alongside `FIREBASE_DB_URL`. |

### Runtime Paths (pre-configured defaults)

These are already set in the Dockerfile and do not need to be overridden in the RunPod template under normal circumstances.

| Variable | Default | Description |
|---|---|---|
| `USE_CUDA` | `1` | Enable CUDA inference. |
| `SAM_WORKER_URL` | `http://127.0.0.1:8001/process-3d` | Internal URL the Main API uses to reach the SAM 3D Worker. |
| `GENERATED_IMAGES_DIR` | `/app/generated/images` | Output directory for generated images. |
| `GENERATED_MODELS_DIR` | `/app/generated/models` | Output directory for generated 3D models. |
| `SAM_TASKS_DIR` | `/app/tasks` | Ticket directory for async job tracking. |
| `SAM_TEMP_DIR` | `/app/temp` | Temporary files during 3D generation. |
| `TRIPO3D_BASE` | `https://api.tripo3d.ai/v2/openapi` | Tripo3D API base URL. |
| `TRIPO3D_MODEL_VERSION` | `v2.0-20240919` | Tripo3D model version string. |
| `TRIPO3D_POLL_SECONDS` | `2.0` | Polling interval for Tripo3D job status. |
| `TRIPO3D_TIMEOUT_SECONDS` | `1800` | Max wait time for a Tripo3D job (30 min). |

---

## GPU Architecture Reference

The Dockerfile compiles CUDA kernels for the following architectures at build time. If you are building for a GPU not in this list, add its compute capability to the `CUDA_ARCH_LIST` and `TORCH_CUDA_ARCH_DOT` build args before building.

```dockerfile
ARG CUDA_ARCH_LIST="90;89;86;80;75"
ARG TORCH_CUDA_ARCH_DOT="9.0;8.9;8.6;8.0;7.5"
```

To target a different architecture, pass it at build time:

```bash
docker build \
  --build-arg HF_TOKEN=hf_... \
  --build-arg CUDA_ARCH_LIST="86" \
  --build-arg TORCH_CUDA_ARCH_DOT="8.6" \
  -f docker/Dockerfile.sam3.runpod \
  -t coarchai-2dto3d:latest \
  .
```

| Compute Capability | GPU Family | Example Models |
|---|---|---|
| 7.5 | Turing | RTX 2080 Ti, T4 |
| 8.0 | Ampere (data center) | A100, A30 |
| 8.6 | Ampere (consumer) | RTX 3090, 3080, A6000 |
| 8.9 | Ada Lovelace | RTX 4090, 4080, L40S |
| 9.0 | Hopper | H100 |

---

## License & Credits

- [Meta SAM 3](https://github.com/facebookresearch/sam3) — Meta AI Research License
- [Meta SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Meta AI Research License
- [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) — Meta AI Research License
- [Tripo3D](https://www.tripo3d.ai) — cloud API, subject to Tripo3D terms of service
- FastAPI / Uvicorn / PyTorch / HuggingFace Transformers / Cloudflare Tunnel
