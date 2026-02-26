# app/services/errors.py
import traceback
from typing import Optional
from fastapi.responses import JSONResponse


def json_error(message: str, stage: str, exc: Optional[BaseException] = None) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": message,
            "stage": stage,
            "trace": traceback.format_exc() if exc else None,
        },
    )