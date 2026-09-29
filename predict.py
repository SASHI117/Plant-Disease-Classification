"""Load the trained MobileNetV2 classifier and predict leaf diseases.

    python predict.py leaf.jpg [--top 3]
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "plant_model_fast.keras"
DEFAULT_CLASSES = ROOT / "models" / "class_names.json"
IMG_SIZE = 160


def pretty(label: str) -> str:
    """'Tomato__Tomato_YellowLeaf__Curl_Virus' -> 'Tomato - YellowLeaf Curl Virus'."""
    crop, _, disease = label.partition("_")
    disease = disease.strip("_").replace("___", " ").replace("__", " ").replace("_", " ")
    disease = disease.removeprefix("bell ").removeprefix(f"{crop} ").strip()
    disease = disease.replace("Spider mites Two spotted spider mite", "Two-spotted spider mite")
    crop = "Pepper (bell)" if crop == "Pepper" else crop
    return f"{crop} - {disease[:1].upper()}{disease[1:]}"


@lru_cache(maxsize=2)
def load(model_path: str = str(DEFAULT_MODEL), classes_path: str = str(DEFAULT_CLASSES)):
    import keras

    model = keras.models.load_model(model_path, compile=False)
    class_names = json.loads(Path(classes_path).read_text())
    n_out = model.output_shape[-1]
    if n_out != len(class_names):
        raise ValueError(f"model has {n_out} outputs but {len(class_names)} class names")
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    """Match training: RGB, 160x160, scaled to [0, 1] (ImageDataGenerator rescale=1/255)."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]


def predict(image: Image.Image, top: int = 3, model_path=None, classes_path=None) -> list[tuple[str, float]]:
    model, class_names = load(str(model_path or DEFAULT_MODEL), str(classes_path or DEFAULT_CLASSES))
    probs = model.predict(preprocess(image), verbose=0)[0]
    order = np.argsort(probs)[::-1][:top]
    return [(class_names[i], float(probs[i])) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args()

    for path in args.images:
        preds = predict(Image.open(path), args.top, args.model)
        print(path.name)
        for label, p in preds:
            print(f"  {p:6.1%}  {pretty(label)}")


if __name__ == "__main__":
    main()
