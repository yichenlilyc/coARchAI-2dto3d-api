# app/routers/firebase_legacy.py
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Body

from app import settings
from app.services.errors import json_error
from app.services.firebase_legacy import fb_fetch, write_png_to_fb
from app.services.common import decode_data_url_to_bytes

router = APIRouter()


# ============================================================
# POST /firebase/sync-to-fbupload
# ============================================================

@router.post("/firebase/sync-to-fbupload")
def firebase_sync_to_fbupload(payload: dict = Body(...)):
    """
    Body:
        { "user": "student_c00", "limit": 200 }
        OR
        { "user": "*", "limit": 200 }   # sync all users

    Behavior:
        - Pull gallery images from Firebase RTDB
        - Convert base64 → PNG
        - Save into FB_UPLOAD_DIR
        - Return public URLs
    """

    try:
        user = payload.get("user")
        limit = int(payload.get("limit", 200))

        if not settings.FIREBASE_DB_URL:
            return json_error("FIREBASE_DB_URL not configured", stage="firebase-sync")

        # ----------------------------------------
        # Determine users to scan
        # ----------------------------------------

        users_to_scan: List[str] = []

        if user in ("*", "all"):
            users_obj = fb_fetch("users")
            if isinstance(users_obj, dict):
                users_to_scan = list(users_obj.keys())
        else:
            if not user:
                return json_error("Missing 'user' in payload", stage="firebase-sync")
            users_to_scan = [user]

        public_base = settings.PUBLIC_BASE_URL.rstrip("/")
        total_items = []

        # ----------------------------------------
        # Fetch gallery entries
        # ----------------------------------------

        for u in users_to_scan:
            path = f"users/{u}/gallery"
            data = fb_fetch(
                path,
                params={
                    "orderBy": json.dumps("$key"),
                    "limitToLast": limit,
                },
            )

            if not isinstance(data, dict):
                continue

            # Sort by key (newest first)
            ordered = sorted(data.items(), key=lambda kv: kv[0], reverse=True)

            for key, val in ordered:
                if not isinstance(val, dict):
                    continue

                img_data = val.get("image")
                if not img_data:
                    continue

                try:
                    raw = decode_data_url_to_bytes(img_data)
                    fname = write_png_to_fb(raw, suggested_name=f"{u}_{key}")
                except Exception:
                    continue

                rel = f"/static/fbupload/{fname}"
                url = f"{public_base}{rel}" if public_base else rel

                total_items.append(
                    {
                        "user": u,
                        "id": key,
                        "name": fname,
                        "url": url,
                        "prompt": val.get("prompt", ""),
                        "timestamp": val.get("timestamp", ""),
                        "version": val.get("version", ""),
                    }
                )

        return {
            "synced": len(total_items),
            "items": total_items,
        }

    except Exception as e:
        return json_error(
            "Firebase sync to fbupload failed",
            stage="firebase-sync",
            exc=e,
        )


# ============================================================
# GET /fbuploads
# ============================================================

@router.get("/fbuploads")
def list_fbuploads(limit: int = 100, offset: int = 0):
    """
    List locally saved Firebase-uploaded images.
    """

    try:
        items = []

        for name in os.listdir(settings.FB_UPLOAD_DIR):
            if name.startswith("."):
                continue

            path = os.path.join(settings.FB_UPLOAD_DIR, name)
            if not os.path.isfile(path):
                continue

            stat = os.stat(path)

            rel = f"/static/fbupload/{name}"
            url = f"{settings.PUBLIC_BASE_URL}{rel}" if settings.PUBLIC_BASE_URL else rel

            items.append(
                {
                    "name": name,
                    "url": url,
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        items.sort(key=lambda x: x["mtime"], reverse=True)

        total = len(items)
        items = items[offset: offset + limit]

        return {
            "count": total,
            "items": items,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        return json_error(
            "Failed to list fbuploads",
            stage="firebase-list",
            exc=e,
        )