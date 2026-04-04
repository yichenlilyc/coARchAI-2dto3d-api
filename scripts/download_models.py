import os
from huggingface_hub import login, hf_hub_download, snapshot_download

# Token injected from Docker build args
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    raise ValueError("ERROR: Could not find HF_TOKEN in the .env variable!")

print("Authenticating with Hugging Face...")
login(token=hf_token)

# --- CONFIGURATION ---
# We use absolute paths to guarantee alignment between API and Worker
BASE_CHECKPOINT_DIR = "/app/checkpoints"
os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)

# 1. Download SAM 2.1 Large
# Target: /app/checkpoints/sam2.1_hiera_large.pt
print("Downloading SAM 2.1 Large...")
hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_large.pt",
    local_dir=BASE_CHECKPOINT_DIR
)

hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_l.yaml",
    local_dir=BASE_CHECKPOINT_DIR
)

# 2. Download SAM 3D Objects
# Target: /app/checkpoints/hf/...
# This matches the path logic in your worker_3d.py
print("Downloading SAM 3D Objects...")
snapshot_download(
    repo_id="facebook/sam-3d-objects",
    local_dir=os.path.join(BASE_CHECKPOINT_DIR, "hf")
)

print("All models successfully downloaded!")
print(f"SAM 2 Weights: {os.path.join(BASE_CHECKPOINT_DIR, 'sam2.1_hiera_large.pt')}")
print(f"SAM 3D Config: {os.path.join(BASE_CHECKPOINT_DIR, 'hf/checkpoints/pipeline.yaml')}")