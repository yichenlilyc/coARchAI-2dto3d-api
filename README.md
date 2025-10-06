# 2Dto3D-Server (TripoSR backend)

Containerized **FastAPI** server that wraps **TripoSR** (and optional shap-e) for single-image → 3D reconstruction.  
Runs on **GPU (CUDA)** or **CPU**, with optional public access via **Cloudflare Tunnel**.

- ✅ One repo, two targets: `app-gpu` and `app-cpu`  
- ✅ Reproducible builds with Docker  
- ✅ Simple health checks & HTTP API  
- ✅ Optional public URLs (Cloudflare Tunnel)

> Tested on Windows 11 + Docker Desktop.  
> Linux works the same (NVIDIA Container Toolkit required for GPU).

---

## Contents

- [Prerequisites](#prerequisites)
- [Repo Layout](#repo-layout)
- [Quick Start (local)](#quick-start-local)
- [API](#api)
- [Config & Models](#config--models)
- [Cloudflare Tunnel (public URLs)](#cloudflare-tunnel-public-urls)
- [Change GPU Architecture](#change-gpu-architecture)
- [Troubleshooting](#troubleshooting)
- [Developing locally (without Docker)](#developing-locally-without-docker)
- [gitignore (suggested)](#gitignore-suggested)
- [License & Credits](#license--credits)
- [Acknowledgements](#acknowledgements)

---

## Prerequisites

### Everyone
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)

### For GPU build
- NVIDIA GPU + latest driver  
- On Windows: Docker Desktop → **Use the WSL 2 based engine**  
- Docker can see your GPU:
```bash
docker run --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi
```

### For CPU build
- No special hardware. It’s slower but works everywhere.

---

## Repo layout
```bash
.
├─ external/TripoSR/
│  ├─ config.yaml           # TripoSR config (required)
│  └─ model.ckpt            # TripoSR weights (required)
├─ scripts/
│  ├─ server.py             # FastAPI app
│  └─ docker/
│     ├─ Dockerfile.gpu
│     └─ Dockerfile.cpu
├─ requirements.txt         # GPU/standard runtime
├─ requirements.cpu.txt     # CPU-only runtime
└─ docker-compose.yml
```
> **Important:** Put your config.yaml and model.ckpt in external/TripoSR/.
> The server reads these by filename via environment variables.

---

## Quick Start (local)
1) Clone & prepare
```bash
git clone https://github.com/YOU/2DTo3D-Server.git
cd 2DTo3D-Server
cp .env.example .env
```
2) Put model files
Place your weights and config here:
```bash
external/TripoSR/model.ckpt
external/TripoSR/config.yaml
```
(These are not committed. See `.gitignore`.)
3) Cloudflare Tunnel (optional, for public access)
Create a tunnel and get a token:
```bash
# One-time on your machine:
cloudflared tunnel create 2dto3d
cloudflared tunnel token 2dto3d
```
Copy the token into `.env`:
```bash
CF_TUNNEL_TOKEN=PASTE_YOUR_TOKEN_HERE
```
Edit `cloudflared/config.yml` and replace hostnames:
```bash
ingress:
  - hostname: api-gpu.YOURDOMAIN.com
    service: http://app-gpu:8000
  - hostname: api-cpu.YOURDOMAIN.com
    service: http://app-cpu:8000
  - service: http_status:404
```
Then create DNS records (in Cloudflare Dashboard or CLI):
```bash
# Example via CLI:
cloudflared tunnel route dns 2dto3d api-gpu.YOURDOMAIN.com
cloudflared tunnel route dns 2dto3d api-cpu.YOURDOMAIN.com
```
4) Run (GPU)
```bash
docker compose --profile gpu up --build -d
# optional public URL:
docker compose --profile tunnel up -d
```
5) Run (CPU)
```bash
docker compose --profile cpu up --build -d
# optional public URL:
docker compose --profile tunnel up -d
```
6) Test locally
- GPU: http://localhost:8000/health
- GPU Server listens on http://localhost:8000
- CPU: http://localhost:8001/health
- CPU Server listens on http://localhost:8001

7) Test via Cloudflare (if enabled)
- GPU: https://api-gpu.YOURDOMAIN.com/health
- CPU: https://api-cpu.YOURDOMAIN.com/health
8) Stop
```bash
docker compose down
# or stop per profile:
docker compose --profile tunnel down
```

---

## API
### Health
**GPU**
```bash
curl http://localhost:8000/health
```
**CPU**
```bash
curl http://localhost:8001/health
```
**Response (example):**
```bash
{
  "ok": true,
  "device": "cuda",
  "cuda_available": true,
  "triposr_available": true,
  "triposr_model_dir": "/app/external/TripoSR",
  "triposr_config": "config.yaml",
  "triposr_weights": "model.ckpt",
  "triposr_config_exists": true,
  "triposr_weights_exists": true
}
```
> Add your inference endpoints to scripts/server.py as needed.

---

## Config & Models
Environment variables (already set in docker-compose.yml):
```bash
TRIPOSR_MODEL_DIR=/app/external/TripoSR
TRIPOSR_CONFIG=config.yaml
TRIPOSR_WEIGHTS=model.ckpt
```
The repo volume-mounts the whole project into the container at `/app`, so you can modify files without rebuilding.

---

## Cloudflare Tunnel (public URLs)
This is optional. Use it if you want to access your API from the internet without opening router ports.
1) Create a tunnel & get a token
- Cloudflare dashboard → Zero Trust → Networks → Tunnels → Create tunnel
- Choose Cloudflared
- Copy the tunnel token (long string, may end with =)
2) Add the token to Docker Compose
Copy the token into `.env`:
```bash
CF_TUNNEL_TOKEN=PASTE_YOUR_TOKEN_HERE
```
Edit `cloudflared/config.yml` and replace hostnames:
```bash
ingress:
  - hostname: api-gpu.YOURDOMAIN.com
    service: http://app-gpu:8000
  - hostname: api-cpu.YOURDOMAIN.com
    service: http://app-cpu:8000
  - service: http_status:404
```
Start it:
```bash
docker compose up -d cloudflared
```
3) Map public hostnames → local services
Cloudflare Zero Trust → Networks → Tunnels → your tunnel → Published application routes (or Hostname routes, depending on UI):
Create two routes:
- api-gpu.yourdomain.com → Type HTTP, URL http://app-gpu:8000
- api-cpu.yourdomain.com → Type HTTP, URL http://app-cpu:8000
Make sure the tunnel status is HEALTHY.
Now test:
```bash
curl https://api-gpu.yourdomain.com/health
curl https://api-cpu.yourdomain.com/health
```
If you see 5xx at first: check that the containers are running and that the service URLs are exactly http://app-gpu:8000 and http://app-cpu:8000 (these are Docker service names on the compose network).

---

## Change GPU Architecture
We pass architecture to NVCC at build time so CUDA kernels are compiled for your GPU.
Edit the GPU build args in `docker-compose.yml`:
```bash
build:
  context: .
  dockerfile: scripts/docker/Dockerfile.gpu
  args:
    CUDA_ARCH_LIST: "120"     # default for RTX 50xx (Blackwell, sm_120)
    TORCH_CUDA_ARCH_DOT: "12.0"
```
**Common values**

| GPU family         | Compute Capability | `CUDA_ARCH_LIST` | Notes                         |
|--------------------|--------------------|------------------|-------------------------------|
| GTX 10xx (Pascal)  | 6.1                | 61               | Old; ensure Torch supports    |
| RTX 20xx (Turing)  | 7.5                | 75               |                               |
| RTX 30xx (Ampere)  | 8.6                | 86               | e.g. 3080/3090               |
| RTX 40xx (Ada)     | 8.9                | 89               | e.g. 4080/4090               |
| RTX 50xx (Blackwell)| 12.0              | 120              | e.g. 5090                    |

If you change these, rebuild: 
```bash
docker compose build app-gpu.
```

---

## Troubleshooting
### GPU container won’t see the GPU
- Docker Desktop → Settings → Resources → enable GPU
- Verify: 
```bash
docker run --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi
```
- Rebuild: 
```bash
docker compose build app-gpu
```
### `torchmcubes` fails to build
- We already pin scikit-build-core, cmake, ninja, and pybind11 in the Dockerfiles. 
  If you add packages, keep those pins or you may hit CMake errors.
- For CPU image, we also install a CPU wheel of torch; mixing CUDA torch into the CPU image will re-introduce CUDA checks.
### Health OK locally but not via Cloudflare
- In the tunnel’s routes, use the Docker service name and port: http://app-gpu:8000, http://app-cpu:8000.
- Check logs: 
```bash
docker compose logs -f cloudflared
```
- Identity/Access: ensure you didn’t enable login/policies while testing.
### Change ports
- GPU is published to host port `8000`, CPU to `8001`.
  Adjust ports: in `docker-compose.yml` if needed.

---

## Developing locally (without Docker)
Requires Python 3.11+ and a working CUDA toolchain for GPU.
```bash
# venv
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
# Pick one:
pip install -r requirements.txt        # GPU / standard
# or
pip install -r requirements.cpu.txt    # CPU-only
# Run
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
```

---

## .gitignore (suggested)
```bsh
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.venv*/
.env

# Editors / OS
.vscode/
.idea/
.DS_Store

# Models and large artifacts
external/TripoSR/*.ckpt
external/TripoSR/checkpoints/
external/TripoSR/outputs/

# Docker
*.log
```

---

## License & Credits
This repo glues together FastAPI + TripoSR + shap-e for containerized serving.
- TripoSR is © its original authors; please follow their license/usage terms for the model and code.
- shap-e is © its original authors; please follow their license/usage terms for the model and code.
- CUDA®, NVIDIA®, and product names are trademarks of their respective owners.

---

## Acknowledgements
Thanks to the open-source community for:
- FastAPI / Uvicorn
- PyTorch / diffusers / transformers
- scikit-build-core, CMake, ninja, pybind11
- Cloudflare Tunnel