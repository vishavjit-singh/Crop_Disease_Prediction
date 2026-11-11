import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402
from conftest import image_bytes  # noqa: E402


@pytest.fixture()
def client(monkeypatch, make_classifier):
    clf, *_ = make_classifier([1.0, 0.0, 0.0], {"Corn": [0.7, 0.1, 0.1, 0.1]})
    monkeypatch.setenv("PRELOAD_MODELS", "0")
    main.state["clf"] = clf
    with TestClient(main.app) as c:
        main.state["clf"] = clf
        yield c
    main.state.clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_crop_and_disease(client):
    r = client.post("/predict", files={"file": ("leaf.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["crop"] == "Corn" and body["disease"] == "Corn Common Rust"
    assert len(body["top_k"]) == 3


def test_rejects_non_images(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/predict", files={"file": ("fake.jpg", b"not really", "image/jpeg")})
    assert r.status_code == 400
