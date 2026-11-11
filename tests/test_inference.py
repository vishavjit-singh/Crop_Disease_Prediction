import numpy as np
import pytest

from app.inference import InvalidImageError, preprocess
from conftest import image_bytes


@pytest.mark.parametrize("size,mode,fmt", [((640, 480), "RGB", "JPEG"), ((100, 300), "L", "PNG"),
                                           ((3081, 897), "RGB", "JPEG"), ((50, 50), "RGBA", "PNG")])
def test_any_image_becomes_model_input(size, mode, fmt):
    batch = preprocess(image_bytes(size, mode, fmt), 256)
    assert batch.shape == (1, 256, 256, 3)
    assert batch.dtype == np.float32
    assert batch.max() > 1.0  # models rescale internally, so pixels stay in 0-255


def test_non_image_rejected():
    with pytest.raises(InvalidImageError):
        preprocess(b"definitely not an image", 256)


def test_router_selects_crop_model(make_classifier):
    clf, router, models = make_classifier([0.1, 0.1, 0.8], {"Rice": [0.05, 0.05, 0.85, 0.05]})
    pred = clf.predict(image_bytes())
    assert pred.crop == "Rice" and pred.crop_confidence == pytest.approx(0.8)
    assert pred.disease == "Rice Leaf Blast" and pred.confidence == pytest.approx(0.85)
    assert models["Rice"].seen is not None and models["Corn"].seen is None
    assert router.seen.shape == (1, 256, 256, 3)


def test_top_k_sorted(make_classifier):
    clf, *_ = make_classifier([0.9, 0.05, 0.05], {"Corn": [0.1, 0.6, 0.25, 0.05]})
    top = clf.predict(image_bytes(), top_k=3).top_k
    assert [t["disease"] for t in top] == ["Corn Gray Leaf Spot", "Corn Healthy", "Corn Common Rust"]


def test_unlabelled_output_does_not_crash(make_classifier):
    # the shipped wheat model has 17 outputs but only 4 class names; v1 raised IndexError here
    wheat = [0.0] * 17
    wheat[9] = 1.0
    clf, *_ = make_classifier([0.0, 1.0, 0.0], {"Wheat": wheat})
    pred = clf.predict(image_bytes())
    assert pred.crop == "Wheat" and pred.labelled is False
    assert "class 9" in pred.disease


def test_missing_crop_model_rejected(labels):
    from app.inference import CropDiseaseClassifier
    from conftest import FakeModel
    with pytest.raises(ValueError, match="Rice"):
        CropDiseaseClassifier(FakeModel([1, 0, 0]), {"Corn": FakeModel([1]), "Wheat": FakeModel([1])}, labels)
