"""Runs the shipped .h5 models on the sample photos. Skipped when TensorFlow/Keras is not installed."""
from pathlib import Path

import pytest

pytest.importorskip("keras")

from app.inference import CropDiseaseClassifier  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture(scope="module")
def clf():
    return CropDiseaseClassifier.from_directory()


@pytest.mark.parametrize("image", sorted(p.name for p in ASSETS.glob("*.jpg")))
def test_real_models_predict_on_samples(clf, image):
    pred = clf.predict((ASSETS / image).read_bytes())
    assert pred.crop in clf.labels.crops
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.top_k and pred.top_k[0]["probability"] == pytest.approx(pred.confidence)
