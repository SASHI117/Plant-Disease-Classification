"""Gradio demo: upload a leaf photo, get the top-3 predicted diseases.

    python app.py            # http://127.0.0.1:7860
    python app.py --share    # temporary public link
"""
import argparse

import gradio as gr

from predict import load, predict, pretty


def classify(image):
    if image is None:
        return {}
    return {pretty(label): p for label, p in predict(image, top=3)}


def build() -> gr.Interface:
    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Leaf image"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Plant Disease Classifier",
        description=(
            "MobileNetV2 fine-tuned on 15 PlantVillage classes (pepper, potato, tomato). "
            "Works best on a single leaf against a plain background, like the training data."
        ),
        flagging_mode="never",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    load()  # load the model before the first request
    build().launch(share=args.share)
