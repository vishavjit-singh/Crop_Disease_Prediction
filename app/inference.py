"""Two-stage inference: a router model identifies the crop, then that crop's model classifies the disease.

This module has no TensorFlow import at module level, so preprocessing and the routing logic can be
tested with lightweight fake models.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger("crop_disease")
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
LABELS_PATH = APP_DIR / "labels.json"


class Model(Protocol):
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray: ...


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable image."""


@dataclass
class Labels:
    image_size: int
    crops: list[str]
    diseases: dict[str, list[str]]
    model_files: dict[str, str]

    @classmethod
    def load(cls, path: Path = LABELS_PATH) -> Labels:
        return cls(**json.loads(path.read_text()))


@dataclass
class Prediction:
    crop: str
    crop_confidence: float
    disease: str
    confidence: float
    top_k: list[dict] = field(default_factory=list)
    labelled: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def preprocess(data: bytes, size: int) -> np.ndarray:
    """Decode any common image format into a (1, size, size, 3) float32 batch in the 0-255 range.

    The saved models include their own Rescaling(1/255) layer, so pixels are not normalised here.
    EXIF rotation is applied, and grayscale, palette and RGBA images are converted to RGB.
    """
    try:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("file is not a readable image") from exc
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)[np.newaxis, ...]


def _label(names: list[str], idx: int, crop: str) -> tuple[str, bool]:
    if idx < len(names):
        return names[idx], True
    # the published wheat model has more output units than documented class names
    log.warning("%s model returned class %d with no label (only %d labels defined)", crop, idx, len(names))
    return f"{crop} class {idx} (unlabelled)", False


class CropDiseaseClassifier:
    def __init__(self, router: Model, crop_models: dict[str, Model], labels: Labels):
        missing = [c for c in labels.crops if c not in crop_models]
        if missing:
            raise ValueError(f"no disease model for: {', '.join(missing)}")
        self.router, self.crop_models, self.labels = router, crop_models, labels

    @classmethod
    def from_directory(cls, model_dir: Path = ROOT / "model", labels: Labels | None = None) -> CropDiseaseClassifier:
        from keras.models import load_model  # heavy import, deferred until models are actually needed

        labels = labels or Labels.load()
        files = labels.model_files
        router = load_model(model_dir / files["router"], compile=False)
        crop_models = {c: load_model(model_dir / files[c], compile=False) for c in labels.crops}
        log.info("loaded router + %d crop models from %s", len(crop_models), model_dir)
        return cls(router, crop_models, labels)

    def predict(self, data: bytes, top_k: int = 3) -> Prediction:
        batch = preprocess(data, self.labels.image_size)
        crop_probs = np.asarray(self.router.predict(batch, verbose=0))[0]
        crop_idx = int(np.argmax(crop_probs))
        crop = self.labels.crops[crop_idx]

        probs = np.asarray(self.crop_models[crop].predict(batch, verbose=0))[0]
        order = np.argsort(probs)[::-1]
        names = self.labels.diseases[crop]
        disease, labelled = _label(names, int(order[0]), crop)
        top = [{"disease": _label(names, int(i), crop)[0], "probability": round(float(probs[i]), 4)}
               for i in order[:top_k]]
        return Prediction(crop=crop, crop_confidence=round(float(crop_probs[crop_idx]), 4), disease=disease,
                          confidence=round(float(probs[order[0]]), 4), top_k=top, labelled=labelled)
