import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import predict  # noqa: E402


def test_class_names_match_keras_folder_order():
    names = json.loads((ROOT / "models" / "class_names.json").read_text())
    assert len(names) == 15
    # flow_from_directory indexes classes by sorted folder name
    assert names == sorted(names)


@pytest.mark.parametrize("label, expected", [
    ("Pepper__bell___Bacterial_spot", "Pepper (bell) - Bacterial spot"),
    ("Potato___healthy", "Potato - Healthy"),
    ("Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato - YellowLeaf Curl Virus"),
])
def test_pretty(label, expected):
    assert predict.pretty(label) == expected


def test_preprocess_matches_training_input():
    img = Image.new("RGBA", (640, 480), (255, 0, 0, 128))
    x = predict.preprocess(img)
    assert x.shape == (1, 160, 160, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_model_loads_and_returns_a_distribution():
    pytest.importorskip("tensorflow")
    model, names = predict.load()
    assert model.output_shape[-1] == len(names)
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))
    top = predict.predict(img, top=15)
    assert len(top) == 15
    assert sum(p for _, p in top) == pytest.approx(1.0, abs=1e-3)
    assert [p for _, p in top] == sorted((p for _, p in top), reverse=True)
