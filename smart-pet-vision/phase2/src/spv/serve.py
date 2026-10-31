"""HTTP service. Run:  SPV_MANIFEST=runs/manifest.json uvicorn spv.serve:app --port 8000

POST /analyze  (multipart field `file`, optional query `images=false`)
GET  /health
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .pipeline import Manifest, PetPipeline

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
app = FastAPI(title="Smart Pet Vision", version="1.0")
_model_lock = Lock()


@lru_cache(maxsize=1)
def pipeline() -> PetPipeline:
    return PetPipeline(Manifest.load(os.environ.get("SPV_MANIFEST", "runs/manifest.json")))


@app.get("/health")
def health():
    with _model_lock:
        m = pipeline().m
    return {"status": "ok", "detector": os.path.basename(m.detector),
            "segmenter": os.path.basename(m.segmenter), "classifier": m.classifier_kind}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), images: bool = Query(True)):
    # Bound application memory even when Content-Length is absent or incorrect.
    data = await file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB.")
    return await run_in_threadpool(_decode_and_analyze, data, images)


def _decode_and_analyze(data: bytes, images: bool):
    """Decode off the event loop; serialize access to the shared model objects."""
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, "Decoded image exceeds 25 million pixels.")
            source.load()
            im = source.copy()
    except Image.DecompressionBombError:
        raise HTTPException(413, "Decoded image dimensions are too large.")
    except (UnidentifiedImageError, OSError, ImportError):
        # ImportError: Ultralytics patches Image.open to try a HEIC decoder that may be absent.
        raise HTTPException(415, "File is not a readable image.")
    with _model_lock:
        return pipeline().analyze(im, with_images=images)
