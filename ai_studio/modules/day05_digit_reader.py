"""Day 5 module - Digit Reader.

Unlocked by: solutions to Day 5 (PyTorch + MNIST).
The notebook exports a TorchScript model:  ai_studio/models/digit_reader.pt
Input contract: float tensor (1, 1, 28, 28), values 0-1, WHITE digit on
BLACK background (MNIST style, ToTensor scaling only - no normalization).
"""
TITLE = "✍️ Digit Reader"
REQUIRES = ["digit_reader.pt"]


def _sketch_to_tensor(sketch):
    """Sketchpad value -> MNIST-style (1,1,28,28) tensor, or None if empty."""
    import numpy as np
    import torch
    from PIL import Image

    img = sketch.get("composite") if isinstance(sketch, dict) else sketch
    if img is None:
        return None, None
    arr = np.asarray(img).astype(np.float32)

    if arr.ndim == 3 and arr.shape[2] == 4:          # RGBA -> flatten on white
        rgb, a = arr[..., :3], arr[..., 3:4] / 255.0
        arr = rgb * a + 255.0 * (1.0 - a)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    # Ink = dark strokes on light background (invert if needed)
    ink = 255.0 - arr if np.median(arr) > 127 else arr
    if ink.max() < 10:                               # empty canvas
        return None, None
    ink = ink / ink.max() * 255.0

    # MNIST-style framing: crop to the drawing, pad square, fit into 20x20,
    # then center inside the 28x28 canvas.
    ys, xs = np.where(ink > 30)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = max(crop.shape)
    sq = np.zeros((side, side), np.float32)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    sq[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    small = Image.fromarray(sq.astype(np.uint8)).resize((20, 20),
                                                        Image.LANCZOS)
    canvas = np.zeros((28, 28), np.float32)
    canvas[4:24, 4:24] = np.asarray(small, dtype=np.float32)

    x = torch.from_numpy(canvas / 255.0).reshape(1, 1, 28, 28)
    preview = np.kron(canvas.astype(np.uint8), np.ones((6, 6), np.uint8))
    return x, preview


def render(models_dir):
    import gradio as gr
    import torch

    model = torch.jit.load(str(models_dir / "digit_reader.pt"),
                           map_location="cpu")
    model.eval()

    def read_digit(sketch):
        x, preview = _sketch_to_tensor(sketch)
        if x is None:
            return {}, None
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        return ({str(i): float(p) for i, p in enumerate(probs)}, preview)

    gr.Markdown(
        "Draw a digit (0-9) — **big and centered**. The network **you trained "
        "on Day 5** reads your handwriting."
    )
    with gr.Row():
        pad = gr.Sketchpad(label="Draw here", canvas_size=(280, 280))
        with gr.Column():
            out = gr.Label(num_top_classes=3, label="I think it's a ...")
            seen = gr.Image(label="What the network sees (28x28)",
                            height=180)

    pad.change(read_digit, pad, [out, seen])
