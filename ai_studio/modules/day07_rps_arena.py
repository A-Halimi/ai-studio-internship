"""Day 7 module - RPS Arena (rock-paper-scissors vs. your own model).

Unlocked by: solutions to Day 7 (transfer learning).
The notebook exports:
  ai_studio/models/rps_arena.pt          TorchScript fine-tuned MobileNetV3
  ai_studio/models/rps_labels.json       class names in training order
Input contract: (1, 3, 224, 224), resize 256 -> center-crop 224,
normalized with ImageNet mean/std.

Webcam note: Gradio's snapshot capture returns None (a known 5+/6 regression) and
`streaming=True` can't share a source with 'upload', so the webcam uses a
streaming tab that keeps the latest frame in a State, and Play reads that State
(never the Image component) — which is what avoids the queue-join 422.
"""
TITLE = "🪨📄✂️ RPS Arena"
REQUIRES = ["rps_arena.pt", "rps_labels.json"]

EMOJI = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def render(models_dir):
    import json
    import random

    import gradio as gr
    import numpy as np
    import torch
    from PIL import Image

    model = torch.jit.load(str(models_dir / "rps_arena.pt"),
                           map_location="cpu")
    model.eval()
    labels = json.loads((models_dir / "rps_labels.json").read_text(
        encoding="utf-8"))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def preprocess(img):
        im = Image.fromarray(img).convert("RGB")
        # Resize short side to 256, center-crop 224 (same as training)
        scale = 256 / min(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)),
                       Image.BILINEAR)
        left = (im.width - 224) // 2
        top = (im.height - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        x = torch.from_numpy(
            np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return ((x - mean) / std).unsqueeze(0)

    def play(img, score, cheat):
        if img is None:
            return ("📷 No frame yet — show your hand to the webcam for a second, "
                    "or use the **Upload** tab.",
                    score, _fmt(score), {})
        try:
            with torch.no_grad():
                probs = torch.softmax(model(preprocess(img))[0], dim=0)
        except Exception as e:
            return f"⚠️ Could not read that image: {e}", score, _fmt(score), {}
        player = labels[int(probs.argmax())]
        conf = {l: float(p) for l, p in zip(labels, probs)}

        if cheat and player in BEATS:
            ai = next(m for m, b in BEATS.items() if b == player)
        else:
            ai = random.choice(list(BEATS))

        if player == ai:
            score["draws"] += 1
            verdict = "🤝 **Draw!**"
        elif BEATS.get(player) == ai:
            score["you"] += 1
            verdict = "🎉 **You win!**"
        else:
            score["ai"] += 1
            verdict = "🤖 **The AI wins!**"

        msg = (f"You played {EMOJI.get(player, '❓')} **{player}** "
               f"(the model is {probs.max():.0%} sure) — "
               f"the AI played {EMOJI.get(ai, '❓')} **{ai}**.\n\n{verdict}")
        return msg, score, _fmt(score), conf

    def _fmt(score):
        return (f"### 🏆 You {score['you']} — {score['ai']} AI "
                f"(draws: {score['draws']})")

    def reset():
        s = {"you": 0, "ai": 0, "draws": 0}
        return s, _fmt(s), "New match — best of luck! 🍀", {}

    gr.Markdown(
        "Rock, paper, scissors against the model **you fine-tuned on Day 7**. "
        "Use the **Webcam** tab (live feed) or the **Upload** tab (opens the "
        "camera on phones), then hit *Play*."
    )
    score_state = gr.State({"you": 0, "ai": 0, "draws": 0})
    wframe = gr.State(None)          # latest live webcam frame (Play reads this, not the Image)
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("📷 Webcam"):
                    webcam = gr.Image(sources=["webcam"], streaming=True,
                                      type="numpy", height=360, label="Show your move!")
                    play_w = gr.Button("Play (webcam) 🎲", variant="primary")
                with gr.Tab("📁 Upload"):
                    upload = gr.Image(sources=["upload"], type="numpy",
                                      height=360, label="Upload your move")
                    play_u = gr.Button("Play (upload) 🎲", variant="primary")
            cheat = gr.Checkbox(
                label="😈 Cheat mode (the AI sees your move first...)",
                value=False)
            rst = gr.Button("Reset match")
        with gr.Column():
            result = gr.Markdown("Show your hand and hit *Play*.")
            scoreboard = gr.Markdown(_fmt({"you": 0, "ai": 0, "draws": 0}))
            conf = gr.Label(num_top_classes=3, label="What the model saw")

    webcam.stream(lambda f, cur: f if f is not None else cur,
                  inputs=[webcam, wframe], outputs=wframe, stream_every=0.25)
    play_w.click(play, [wframe, score_state, cheat],
                 [result, score_state, scoreboard, conf])
    play_u.click(play, [upload, score_state, cheat],
                 [result, score_state, scoreboard, conf])
    rst.click(reset, None, [score_state, scoreboard, result, conf])
