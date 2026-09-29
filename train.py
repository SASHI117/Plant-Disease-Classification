"""Reproduce the notebook training run outside Colab.

    kaggle datasets download -d emmarex/plantdisease && unzip -q plantdisease.zip -d data
    python train.py --source data/PlantVillage --work data/split

Two phases, as in the notebook: train a new head on a frozen MobileNetV2 for
3 epochs (lr 1e-3), then unfreeze the last 40 layers and fine-tune for 4
epochs (lr 1e-4).
"""
import argparse
import json
import random
import shutil
from pathlib import Path

IMG_SIZE = 160
IMG_EXT = {".jpg", ".jpeg", ".png"}


def split_dataset(source: Path, work: Path, val_frac: float = 0.2, seed: int = 42) -> tuple[Path, Path]:
    """Per-class split into work/train and work/valid (idempotent)."""
    train_dir, valid_dir = work / "train", work / "valid"
    if train_dir.exists() and valid_dir.exists():
        return train_dir, valid_dir
    rng = random.Random(seed)
    for cls in sorted(p for p in source.iterdir() if p.is_dir()):
        images = sorted(p for p in cls.iterdir() if p.suffix.lower() in IMG_EXT)
        rng.shuffle(images)
        n_val = int(len(images) * val_frac)
        for subset, files in (("valid", images[:n_val]), ("train", images[n_val:])):
            dest = work / subset / cls.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
    return train_dir, valid_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="folder with one sub-folder per class")
    ap.add_argument("--work", type=Path, default=Path("data/split"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=4)
    ap.add_argument("--unfreeze", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir, valid_dir = split_dataset(args.source, args.work)

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=25, width_shift_range=0.15,
        height_shift_range=0.15, zoom_range=0.15, horizontal_flip=True,
    )
    valid_gen = ImageDataGenerator(rescale=1.0 / 255)
    flow = dict(target_size=(IMG_SIZE, IMG_SIZE), batch_size=args.batch, class_mode="categorical")
    train_data = train_gen.flow_from_directory(train_dir, **flow)
    valid_data = valid_gen.flow_from_directory(valid_dir, shuffle=False, **flow)

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(train_data.num_classes, activation="softmax")(x)
    model = Model(base.input, out)

    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_data, validation_data=valid_data, epochs=args.head_epochs)

    for layer in base.layers[-args.unfreeze:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_data, validation_data=valid_data, epochs=args.finetune_epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "plant_model_fast.keras")
    class_names = sorted(train_data.class_indices, key=train_data.class_indices.get)
    (args.out / "class_names.json").write_text(json.dumps(class_names, indent=2))
    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved model, class names and history to {args.out}/")


if __name__ == "__main__":
    main()
