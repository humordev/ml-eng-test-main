from __future__ import annotations

from io import BytesIO
from typing import Annotated

import cv2
import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from app.inference import WallDetector, build_wall_mask
from app.postprocess import detect_rooms_from_walls
from app.schemas import InferenceResponse, InferenceType
from app.utils import draw_rooms, draw_walls, encode_image_base64, rooms_to_schema, walls_to_schema

app = FastAPI(title="Blueprint CV Inference API", version="0.1.0")

wall_detector = WallDetector(model_dir="models")


def _load_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    content_type = upload.content_type or ""

    if content_type == "application/pdf":
        return _load_pdf_first_page_to_bgr(content)

    return _load_image_bytes_to_bgr(content)


def _load_image_bytes_to_bgr(content: bytes) -> np.ndarray:
    try:
        pil = Image.open(BytesIO(content)).convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def _load_pdf_first_page_to_bgr(content: bytes) -> np.ndarray:
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.page_count == 0:
                raise HTTPException(status_code=400, detail="PDF has no pages.")
            page = document.load_page(0)
            pix = page.get_pixmap(dpi=300, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "wall_detector_backend": wall_detector.get_backend()}


@app.post("/run-inference", response_model=InferenceResponse)
def run_inference(
    image: Annotated[UploadFile, File(...)],
    type: Annotated[InferenceType, Query(description="Inference type")],
) -> InferenceResponse:
    if image.content_type is None:
        raise HTTPException(status_code=400, detail="Could not detect upload content type.")
    if not image.content_type.startswith("image/") and image.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image or PDF.")

    bgr = _load_upload_to_bgr(image)
    walls = wall_detector.predict_walls(bgr)
    wall_mask = build_wall_mask(bgr.shape[:2], walls, thickness=4)

    if type == "wall":
        wall_image = draw_walls(bgr, walls)
        return InferenceResponse(
            type=type,
            annotated_image_base64=encode_image_base64(wall_image),
            wall_count=len(walls),
            walls=walls_to_schema(walls),
            message="Wall inference completed.",
        )

    if type == "room":
        rooms = detect_rooms_from_walls(wall_mask=wall_mask)
        room_image = draw_rooms(draw_walls(bgr, walls), rooms)
        return InferenceResponse(
            type=type,
            annotated_image_base64=encode_image_base64(room_image),
            wall_count=len(walls),
            room_count=len(rooms),
            walls=walls_to_schema(walls),
            rooms=rooms_to_schema(rooms),
            message="Room inference completed.",
        )

    # Placeholders for additional tasks listed in the challenge.
    passthrough = draw_walls(bgr, walls)
    return InferenceResponse(
        type=type,
        annotated_image_base64=encode_image_base64(passthrough),
        wall_count=len(walls),
        walls=walls_to_schema(walls),
        message=f"'{type}' is currently a placeholder that returns wall detections.",
    )
