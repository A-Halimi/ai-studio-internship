"""Day 6 module - Photo Detective.

Unlocked by: solutions to Day 6 (convolutional neural networks).
The notebook exports:
  ai_studio/models/photo_detective.pt            TorchScript CNN
  ai_studio/models/photo_detective_classes.json  list of 10 class names
Input contract: float tensor (1, 3, 32, 32), normalized with
mean=0.5 / std=0.5 per channel ( (x - 0.5) / 0.5 ), CIFAR-10 classes.
"""
TITLE = "🕵️ Photo Detective"
REQUIRES = ["photo_detective.pt", "photo_detective_classes.json"]


def render(models_dir):
    import json

    import gradio as gr
    import numpy as np
    import torch
    from PIL import Image

    model = torch.jit.load(str(models_dir / "photo_detective.pt"),
                           map_location="cpu")
    model.eval()
    classes = json.loads(
        (models_dir / "photo_detective_classes.json").read_text(
            encoding="utf-8"))

    def investigate(img):
        if img is None:
            return {}, None
        im = Image.fromarray(img).convert("RGB")
        # Square center-crop, then shrink to the 32x32 world of CIFAR-10
        side = min(im.size)
        left = (im.width - side) // 2
        top = (im.height - side) // 2
        im32 = im.crop((left, top, left + side, top + side)).resize(
            (32, 32), Image.LANCZOS)

        x = torch.from_numpy(
            np.asarray(im32, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        x = (x - 0.5) / 0.5
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)

        preview = np.kron(np.asarray(im32),
                          np.ones((8, 8, 1), np.uint8))
        return ({c: float(p) for c, p in zip(classes, probs)}, preview)

    gr.Markdown(
        "Upload a photo of one of these: **" + ", ".join(classes) + "**. "
        "The CNN **you trained on Day 6** investigates. It only sees a tiny "
        "32x32 version — just like it did in training!"
    )
    with gr.Row():
        inp = gr.Image(type="numpy", label="Evidence (your photo)")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Verdict")
            seen = gr.Image(label="What the detective sees (32x32)",
                            height=200)

    inp.change(investigate, inp, [out, seen])
