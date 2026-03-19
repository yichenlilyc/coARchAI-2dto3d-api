import subprocess
import os
import sys
import time

print("Starting All-in-One Multi-GPU Launcher...", flush=True)

processes = []

# 1. Start Cloudflare Tunnel (if token is provided in RunPod UI)
cf_token = os.environ.get("CF_TUNNEL_TOKEN")
if cf_token:
    print("Detected CF_TUNNEL_TOKEN. Starting Cloudflare Tunnel...", flush=True)
    cf_process = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", cf_token])
    processes.append(cf_process)
else:
    print("No CF_TUNNEL_TOKEN found. Skipping Cloudflare.", flush=True)

# 2. Start Main API (Force it to only see GPU 0)
env_main = os.environ.copy()
env_main["CUDA_VISIBLE_DEVICES"] = "0"
print("Launching Main API on GPU 0 (Port 8000)...", flush=True)
api_process = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"], 
    env=env_main
)
processes.append(api_process)

time.sleep(2)

# 3. Start SAM Worker using the Conda Python (Force it to only see GPU 1)
env_worker = os.environ.copy()
env_worker["CUDA_VISIBLE_DEVICES"] = "1"
print("Launching SAM 3D Worker on GPU 1 (Port 8001)...", flush=True)
worker_process = subprocess.Popen([
    "/root/miniconda3/envs/sam3d-objects/bin/python", 
    "-m", "uvicorn", "app.services.sam3d_worker:app", "--host", "0.0.0.0", "--port", "8001"
], env=env_worker)
processes.append(worker_process)

try:
    # Wait for all background processes
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("Shutting down all processes...")
    for p in processes:
        p.terminate()
    sys.exit(0)