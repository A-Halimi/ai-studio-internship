"""Day 3 module - Pattern Finder.

Unlocked by: solutions to Day 3 (unsupervised learning).
The notebook exports an unlock token:  ai_studio/models/pattern_finder.json
    {"default_k": 5, "made_by": "<student name>"}

The clustering itself runs live: k-means groups every pixel of an uploaded
photo into k color clusters - exactly what the students did in the notebook.
"""
TITLE = "🎨 Pattern Finder"
REQUIRES = ["pattern_finder.json"]


def render(models_dir):
    import json

    import gradio as gr
    import numpy as np
    from PIL import Image

    cfg = json.loads((models_dir / "pattern_finder.json").read_text(
        encoding="utf-8"))
    default_k = int(cfg.get("default_k", 5))

    def find_palette(img, k):
        if img is None:
            return None, None
        from sklearn.cluster import KMeans

        im = Image.fromarray(img).convert("RGB")
        small = im.copy()
        small.thumbnail((260, 260))
        pixels = np.asarray(small).reshape(-1, 3).astype(np.float64)

        km = KMeans(n_clusters=int(k), n_init=4, random_state=0).fit(pixels)
        centers = km.cluster_centers_.astype(np.uint8)
        counts = np.bincount(km.labels_, minlength=int(k))
        order = np.argsort(-counts)

        # Palette bar, widest cluster first
        bar = np.zeros((80, 400, 3), np.uint8)
        x = 0
        for idx in order:
            w = int(round(counts[idx] / counts.sum() * 400))
            bar[:, x:x + w] = centers[idx]
            x += w
        if x < 400:
            bar[:, x:] = centers[order[-1]]

        # The photo repainted with only k colors
        quant = centers[km.labels_].reshape(np.asarray(small).shape)
        return bar, quant

    gr.Markdown(
        "Upload any photo. K-means — the same algorithm **you used on Day 3** "
        "— finds its *k* dominant colors and repaints the image with only "
        "those colors."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Your photo")
            k = gr.Slider(2, 12, value=default_k, step=1,
                          label="k (number of color clusters)")
            btn = gr.Button("Find the palette 🎨", variant="primary")
        with gr.Column():
            palette = gr.Image(label="Dominant colors")
            quantized = gr.Image(label=f"Repainted with only k colors")

    btn.click(find_palette, [inp, k], [palette, quantized])
    k.release(find_palette, [inp, k], [palette, quantized])
