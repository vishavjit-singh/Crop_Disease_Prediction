"""FastAPI service for crop disease prediction.

Run: uvicorn app.main:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import CropDiseaseClassifier, InvalidImageError

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")

state: dict[str, CropDiseaseClassifier] = {}


def get_classifier() -> CropDiseaseClassifier:
    if "clf" not in state:
        state["clf"] = CropDiseaseClassifier.from_directory()
    return state["clf"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("PRELOAD_MODELS", "1") == "1":
        get_classifier()  # load once at startup instead of on the first request
    yield
    state.clear()


app = FastAPI(
    title="Crop Disease Prediction API",
    version="2.0.0",
    description="Identifies the crop (corn, wheat, rice) in a leaf photo, then classifies its disease.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": "clf" in state}


@app.get("/ping", include_in_schema=False)
def ping() -> str:  # kept for clients of v1
    return "Hello"


@app.get("/labels")
def labels() -> dict:
    lab = get_classifier().labels
    return {"crops": lab.crops, "diseases": lab.diseases}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"expected an image upload, got {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        result = get_classifier().predict(data, top_k=max(1, min(top_k, 5)))
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
