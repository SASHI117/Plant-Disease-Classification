"""Download a small, class-balanced sample from the original PlantVillage
release (github.com/spMohanty/PlantVillage-Dataset, raw/color) into folders
named like the training classes, for an evaluation outside the Kaggle copy.

    python scripts/fetch_plantvillage_sample.py --per-class 10 --out data/pv_sample
    python evaluate.py data/pv_sample --out reports/pv_sample
"""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"

# original folder name -> training class name (Kaggle emmarex/plantdisease)
CLASSES = {
    "Pepper,_bell___Bacterial_spot": "Pepper__bell___Bacterial_spot",
    "Pepper,_bell___healthy": "Pepper__bell___healthy",
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Tomato___Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot": "Tomato__Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato__Tomato_mosaic_virus",
    "Tomato___healthy": "Tomato_healthy",
}


def retry(fn, attempts=5):
    for i in range(attempts):
        try:
            return fn()
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "plantvillage-sample"})
    return retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/pv_sample"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Directory SHAs from the (small) parent listing, then one tree call per
    # class: the contents API refuses directories with more than 1000 files.
    shas = {d["name"]: d["sha"] for d in get_json(f"{API}/contents/raw/color")}
    for k, (src, dst) in enumerate(CLASSES.items()):
        files = [t["path"] for t in get_json(f"{API}/git/trees/{shas[src]}")["tree"] if t["type"] == "blob"]
        out = args.out / dst
        out.mkdir(parents=True, exist_ok=True)
        for name in random.Random(args.seed + k).sample(files, args.per_class):
            target = out / name.replace(" ", "_")
            if not target.exists():
                url = f"{RAW}/{urllib.parse.quote(src)}/{urllib.parse.quote(name)}"
                retry(lambda url=url, target=target: urllib.request.urlretrieve(url, target))
        print(f"{dst}: {args.per_class} of {len(files)}")


if __name__ == "__main__":
    main()
