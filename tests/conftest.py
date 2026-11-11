from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.inference import CropDiseaseClassifier, Labels


class FakeModel:
    """Returns a fixed probability vector and records the batch it was given."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)
        self.seen = None

    def predict(self, batch, verbose=0):
        self.seen = batch
        return self.probs[np.newaxis, :]


def image_bytes(size=(640, 480), mode="RGB", fmt="JPEG"):
    buf = BytesIO()
    Image.new(mode, size, color=(30, 160, 60) if mode == "RGB" else 120).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def labels():
    return Labels.load()


@pytest.fixture()
def make_classifier(labels):
    def _make(router_probs, crop_probs):
        router = FakeModel(router_probs)
        models = {c: FakeModel(crop_probs.get(c, [1, 0, 0, 0])) for c in labels.crops}
        return CropDiseaseClassifier(router, models, labels), router, models
    return _make
