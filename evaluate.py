"""Per-class evaluation of the trained model on a held-out folder.

    python evaluate.py data/split/valid --out reports/

Writes metrics.json (accuracy, macro-F1, per-class precision/recall/F1) and
confusion_matrix.png. Overall accuracy hides which diseases get confused
with each other, e.g. early vs late blight, which is what matters in the field.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from predict import DEFAULT_MODEL, IMG_SIZE, load, pretty


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="folder with one sub-folder per class")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model, class_names = load(str(args.model))
    data = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        args.data, target_size=(IMG_SIZE, IMG_SIZE), batch_size=64,
        class_mode="categorical", shuffle=False,
    )
    folder_classes = sorted(data.class_indices, key=data.class_indices.get)
    if folder_classes != class_names:
        raise SystemExit("class folders do not match models/class_names.json")

    y_true = data.classes
    y_pred = np.argmax(model.predict(data, verbose=1), axis=1)

    labels = [pretty(c) for c in class_names]
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, digits=4)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    n = len(labels)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= 0.005:   # blank zeros so the confusions stand out
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(n), [str(i + 1) for i in range(n)])
    ax.set_yticks(range(n), [f"{i + 1}  {lbl}" for i, lbl in enumerate(labels)])
    ax.set_xlabel("Predicted class (numbered as rows)")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix, row-normalised (n = {len(y_true)})", loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print(f"accuracy {report['accuracy']:.4f}  macro-F1 {report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
