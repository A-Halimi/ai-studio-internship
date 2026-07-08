"""Day 7 module - RPS Arena (rock-paper-scissors vs. your own model).

Unlocked by: solutions to Day 7 (transfer learning).
The notebook exports:
  ai_studio/models/rps_arena.pt          TorchScript fine-tuned MobileNetV3
  ai_studio/models/rps_labels.json       class names in training order
Input contract: (1, 3, 224, 224), resize 256 -> center-crop 224,
normalized with ImageNet mean/std.
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
            return ("📷 No photo yet — click the webcam **capture** button, or use "
                    "the **Upload** tab (opens the camera on phones).",
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
        "Give it a hand photo — **Upload** is the reliable path (on a phone it "
        "opens the camera); the **Webcam** tab works too when your browser allows it."
    )
    score_state = gr.State({"you": 0, "ai": 0, "draws": 0})
    with gr.Row():
        with gr.Column():
            cam = gr.Image(sources=["upload", "webcam"], type="numpy",
                           label="Show your move!")
            cheat = gr.Checkbox(
                label="😈 Cheat mode (the AI sees your move first...)",
                value=False)
            with gr.Row():
                btn = gr.Button("Play this move! 🎲", variant="primary")
                rst = gr.Button("Reset match")
        with gr.Column():
            result = gr.Markdown("Show your hand to the camera and hit *Play*.")
            scoreboard = gr.Markdown(_fmt({"you": 0, "ai": 0, "draws": 0}))
            conf = gr.Label(num_top_classes=3, label="What the model saw")

    btn.click(play, [cam, score_state, cheat],
              [result, score_state, scoreboard, conf])
    rst.click(reset, None, [score_state, scoreboard, result, conf])
