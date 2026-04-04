# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app import settings
from app.routers.health import router as health_router
from app.routers.generated import router as generated_router
from app.routers.firebase_legacy import router as firebase_legacy_router
from app.routers.legacy_image_to_3d import router as legacy_image_to_3d_router
from app.routers.uploads import router as uploads_router
from app.routers.presets import router as presets_router
from app.routers.sam import router as sam_router
from app.routers.dictation import router as dictation_router


def create_app() -> FastAPI:
    server_configs = []

    # Check if the environment variable is set already, if so use it
    public_url = os.environ.get("PUBLIC_BASE_URL")
    if public_url:
        server_configs.append({"url": public_url, "description": "Public Cloudflare Tunnel"})

    # Pass the metadata and dynamic server list to FastAPI
    app = FastAPI(
        title="CoARchAI 2D-to-3D API",
        description="Image to 3D Generation API (Shap-E, Tripo, Meta SAM)",
        version="0.1.0",
        servers=server_configs
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static mounts
    app.mount("/static/generated/images", StaticFiles(directory=settings.GENERATED_IMAGES_DIR), name="generated_images_static")
    app.mount("/static/generated/models", StaticFiles(directory=settings.GENERATED_MODELS_DIR), name="generated_models_static")
    app.mount("/static/fbupload", StaticFiles(directory=settings.FB_UPLOAD_DIR), name="fbupload_static")

    app.mount("/presets/maps",StaticFiles(directory=settings.PRESET_MAPS_DIR),name="preset_maps_static")
    app.mount("/presets/models",StaticFiles(directory=settings.PRESET_MODELS_DIR),name="preset_models_static")
    app.mount("/static/uploads/models",StaticFiles(directory=settings.UPLOAD_MODELS_DIR),name="upload_models_static")



    # Routers
    app.include_router(health_router, tags=["health"])
    app.include_router(generated_router, tags=["generated"])
    app.include_router(firebase_legacy_router, tags=["legacy"])
    app.include_router(legacy_image_to_3d_router, tags=["legacy"])
    app.include_router(uploads_router, tags=["uploads"])
    app.include_router(presets_router, tags=["presets"])
    app.include_router(sam_router, tags=["sam"])
    app.include_router(dictation_router, tags=["dictation"])

    return app


app = create_app()