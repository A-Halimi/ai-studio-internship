# =====================================================================
#  The Lab · AI Studio — Intro to AI
#  Author: Dr. Abdelghafour HALIMI  |  https://ahalimi.com
#  Copyright (c) 2026 Dr. Abdelghafour HALIMI. All rights reserved.
# =====================================================================
"""Export a fully precomputed simulation of the AI Studio app for the website.

Runs every trained Studio model over a fixed set of sample inputs and writes
`studio_sim/studio.json` + `studio_sim/img/*` so the site can replay the app
without any Python runtime. Deterministic (fixed seeds, fixed file names).

Run inside the course image with the repo mounted at /workspace:
  docker run --rm --gpus all --ipc=host -v "${PWD}:/workspace" -w /workspace \
    abdelghafour1/ai-studio:latest python scripts/export_studio_sim.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil

import numpy as np
import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "ai_studio" / "models"
DATA = pathlib.Path(os.environ.get("COURSE_DATA_DIR", REPO / "data"))
SAMPLES = REPO / "data_samples"
OUT = REPO / "studio_sim"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

manifest: dict = {"schemaVersion": 1, "courseId": "ai-studio"}


def save_png(arr: np.ndarray, name: str) -> str:
    Image.fromarray(arr.astype(np.uint8)).save(IMG / name)
    return f"img/{name}"


def probs_dict(labels, probs, nd=4):
    return {str(l): round(float(p), nd) for l, p in zip(labels, probs)}


# ── 🐧 Prediction Machine: export the whole k-NN so the site runs it live ──
print("• prediction machine")
import joblib  # noqa: E402

bundle = joblib.load(MODELS / "prediction_machine.joblib")
pipe = bundle["model"]
scaler = pipe.named_steps["standardscaler"]
knn = pipe.named_steps["kneighborsclassifier"]
manifest["predictionMachine"] = {
    "featureNames": list(bundle["feature_names"]),
    "classes": [str(c) for c in knn.classes_],
    "k": int(knn.n_neighbors),
    "scalerMean": [round(float(v), 6) for v in scaler.mean_],
    "scalerScale": [round(float(v), 6) for v in scaler.scale_],
    "points": [[round(float(v), 4) for v in row] for row in knn._fit_X],
    "labels": [int(v) for v in knn._y],
    "sliders": [
        {"label": "Bill length (mm)", "min": 30, "max": 60, "step": 0.1, "value": 44},
        {"label": "Bill depth (mm)", "min": 13, "max": 22, "step": 0.1, "value": 17},
        {"label": "Flipper length (mm)", "min": 170, "max": 235, "step": 1, "value": 200},
        {"label": "Body mass (g)", "min": 2700, "max": 6300, "step": 50, "value": 4200},
    ],
}

# ── 🎨 Pattern Finder: config only — the site runs k-means itself ──
print("• pattern finder")
pf_cfg = json.loads((MODELS / "pattern_finder.json").read_text(encoding="utf-8"))
manifest["patternFinder"] = {"defaultK": int(pf_cfg.get("default_k", 5)), "minK": 2, "maxK": 12}

# ── ✍️ Digit Reader: 10 MNIST test digits through the real TorchScript net ──
print("• digit reader")
from torchvision import datasets  # noqa: E402

digit_model = torch.jit.load(str(MODELS / "digit_reader.pt"), map_location="cpu")
digit_model.eval()
mnist = datasets.MNIST(str(DATA), train=False, download=False)
digit_samples = []
seen: set[int] = set()
for img, label in mnist:
    if label in seen:
        continue
    seen.add(label)
    arr = np.asarray(img, dtype=np.float32)  # 28x28, 0..255, white on black
    x = torch.from_numpy(arr / 255.0).reshape(1, 1, 28, 28)
    with torch.no_grad():
        probs = torch.softmax(digit_model(x)[0], dim=0)
    big = np.kron(arr.astype(np.uint8), np.ones((6, 6), np.uint8))
    digit_samples.append({
        "id": f"digit_{label}",
        "label": int(label),
        "img": save_png(big, f"digit_{label}.png"),
        "probs": probs_dict(range(10), probs),
    })
    if len(seen) == 10:
        break
digit_samples.sort(key=lambda s: s["label"])
manifest["digitReader"] = {"samples": digit_samples}

# ── 🕵️ Photo Detective: the golden retriever + one CIFAR-10 test image per class ──
print("• photo detective")
det_model = torch.jit.load(str(MODELS / "photo_detective.pt"), map_location="cpu")
det_model.eval()
det_classes = json.loads((MODELS / "photo_detective_classes.json").read_text(encoding="utf-8"))


def investigate(im: Image.Image):
    """Exactly the module's preprocessing: center-crop square -> 32x32 -> (x-.5)/.5."""
    im = im.convert("RGB")
    side = min(im.size)
    left = (im.width - side) // 2
    top = (im.height - side) // 2
    im32 = im.crop((left, top, left + side, top + side)).resize((32, 32), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im32, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    x = (x - 0.5) / 0.5
    with torch.no_grad():
        probs = torch.softmax(det_model(x)[0], dim=0)
    preview = np.kron(np.asarray(im32), np.ones((8, 8, 1), np.uint8))
    return probs, preview


det_samples = []
dog = Image.open(SAMPLES / "gallery" / "golden_retriever.jpg")
probs, preview = investigate(dog)
dog.save(IMG / "dog_photo.jpg", quality=88)
det_samples.append({
    "id": "dog_photo",
    "name": "Golden retriever · real photo",
    "thumb": "img/dog_photo.jpg",
    "preview": save_png(preview, "dog_photo_32.png"),
    "probs": probs_dict(det_classes, probs),
})
cifar = datasets.CIFAR10(str(DATA), train=False, download=False)
picked: set[int] = set()
for img, label in cifar:
    if label in picked:
        continue
    picked.add(label)
    cls = det_classes[label]
    probs, preview = investigate(img)
    det_samples.append({
        "id": f"cifar_{cls}",
        "name": f"{cls.capitalize()} · CIFAR-10 test set",
        "thumb": save_png(preview, f"cifar_{cls}.png"),
        "preview": f"img/cifar_{cls}.png",
        "probs": probs_dict(det_classes, probs),
    })
    if len(picked) == 10:
        break
manifest["photoDetective"] = {"classes": det_classes, "samples": det_samples}

# ── ✂️ RPS Arena: two sample photos per move through the fine-tuned MobileNet ──
print("• rps arena")
rps_model = torch.jit.load(str(MODELS / "rps_arena.pt"), map_location="cpu")
rps_model.eval()
rps_labels = json.loads((MODELS / "rps_labels.json").read_text(encoding="utf-8"))
_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def rps_probs(im: Image.Image):
    im = im.convert("RGB")
    scale = 256 / min(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    left = (im.width - 224) // 2
    top = (im.height - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = ((x - _mean) / _std).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(rps_model(x)[0], dim=0)


rps_samples = []
for move in ("rock", "paper", "scissors"):
    for n in ("01", "07"):
        src = SAMPLES / "rps_starter" / move / f"{move}_{n}.jpg"
        im = Image.open(src)
        shutil.copyfile(src, IMG / f"rps_{move}_{n}.jpg")
        rps_samples.append({
            "id": f"rps_{move}_{n}",
            "move": move,
            "thumb": f"img/rps_{move}_{n}.jpg",
            "probs": probs_dict(rps_labels, rps_probs(im)),
        })
manifest["rpsArena"] = {"labels": rps_labels, "samples": rps_samples}

# ── 💬 Language Lab: every preset through the real transformer pipelines ──
print("• language lab (transformers)")
from transformers import pipeline  # noqa: E402

ll_cfg = json.loads((MODELS / "language_lab.json").read_text(encoding="utf-8"))
starters = ll_cfg.get("story_starters") or []
default_temp = float(ll_cfg.get("default_temperature", 0.9))

mood_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
MOOD_PRESETS = [
    "This course is the best thing ever!",
    "The demo worked on the first try!",
    "Penguins are absolutely fascinating animals.",
    "I dropped my ice cream and now the day is ruined.",
    "My model predicted everything wrong today.",
    "Training for six hours and the loss went up.",
]
mood_out = []
for text in MOOD_PRESETS:
    r = mood_pipe(text)[0]
    other = "NEGATIVE" if r["label"] == "POSITIVE" else "POSITIVE"
    mood_out.append({"text": text, "scores": {r["label"]: round(float(r["score"]), 4),
                                              other: round(1 - float(r["score"]), 4)}})

gen_pipe = pipeline("text-generation", model="gpt2")
TEMPS = [0.4, 0.9, 1.3]
gens: dict[str, list[str]] = {}
for si, prompt in enumerate(starters):
    for ti, temp in enumerate(TEMPS):
        outs = []
        for seed in range(2):
            torch.manual_seed(1000 * si + 100 * ti + seed)
            out = gen_pipe(prompt, max_new_tokens=60, do_sample=True, temperature=temp,
                           top_p=0.95, pad_token_id=50256)
            outs.append(out[0]["generated_text"])
        gens[f"{si}-{ti}"] = outs

fill_pipe = pipeline("fill-mask", model="distilbert-base-uncased")
FILL_PRESETS = [
    "Students in the lab love to ___ every day.",
    "The robot learned to ___ all by itself.",
    "On Demo Day everyone wants to ___ their Studio.",
    "A neural network is really just a pile of ___.",
    "My favorite thing about AI is that it can ___.",
]
fill_out = []
for sentence in FILL_PRESETS:
    masked = sentence.replace("___", fill_pipe.tokenizer.mask_token, 1)
    guesses = {r["token_str"].strip(): round(float(r["score"]), 4) for r in fill_pipe(masked)[:5]}
    fill_out.append({"sentence": sentence, "guesses": guesses})

zs_pipe = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
SORT_PRESETS = [
    ("The team trained a neural network on GPUs.", "science, sports, cooking, music"),
    ("The goalkeeper saved a penalty in the final minute.", "science, sports, cooking, music"),
    ("Add two cups of flour and bake for twenty minutes.", "science, sports, cooking, music"),
    ("The orchestra tuned their violins before the concert.", "science, sports, cooking, music"),
]
sort_out = []
for text, labels_csv in SORT_PRESETS:
    labels = [l.strip() for l in labels_csv.split(",")]
    r = zs_pipe(text, candidate_labels=labels)
    sort_out.append({"text": text, "labels": labels_csv,
                     "scores": {l: round(float(s), 4) for l, s in zip(r["labels"], r["scores"])}})

manifest["languageLab"] = {
    "starters": starters,
    "defaultTemperature": default_temp,
    "temps": TEMPS,
    "story": gens,
    "mood": mood_out,
    "fill": fill_out,
    "sort": sort_out,
}

# ── 🎮 Game Master: real episode replays rendered to GIFs ──
print("• game master (gym episodes)")
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def save_gif(frames, name, fps, width=420):
    small = []
    for f in frames:
        im = Image.fromarray(f)
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.BILINEAR)
        small.append(np.asarray(im))
    imageio.mimsave(IMG / name, small, fps=fps, loop=0)
    return f"img/{name}"


q_table = np.load(MODELS / "game_master_qtable.npy")
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")
state, _ = env.reset(seed=0)
frames = [env.render()]
total = 0.0
for _ in range(30):
    state, reward, term, trunc, _ = env.step(int(np.argmax(q_table[state])))
    frames.append(env.render())
    total += reward
    if term or trunc:
        break
env.close()
fl_msg = ("🏆 **The agent reached the gift!** It follows the Q-table it learned — no luck involved."
          if total > 0 else "🕳️ Oops — it fell in a hole. Retrain it in the notebook!")
frozenlake = {"gif": save_gif(frames, "frozenlake.gif", fps=3, width=360),
              "message": fl_msg, "steps": len(frames) - 1}


def cartpole_run(trained: bool):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = None
    if trained:
        from stable_baselines3 import DQN
        model = DQN.load(str(MODELS / "game_master_dqn.zip"), device="cpu")
    obs, _ = env.reset(seed=42)
    frames, total = [], 0.0
    for step in range(500):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        if step % 2 == 0:
            frames.append(env.render())
        if term or trunc:
            break
    env.close()
    who = "your trained DQN" if trained else "an untrained random agent"
    return {"gif": save_gif(frames, f"cartpole_{'dqn' if trained else 'random'}.gif", fps=25),
            "steps": int(total),
            "message": f"**{who}** balanced the pole for **{int(total)} steps** (500 = perfect)."}


cartpole = {"random": cartpole_run(False), "dqn": cartpole_run(True)}
manifest["gameMaster"] = {"frozenlake": frozenlake, "cartpole": cartpole}

# ── shared photo gallery for Pattern Finder (real photos only) ──
manifest["gallery"] = [
    {"id": "dog_photo", "name": "Golden retriever", "img": "img/dog_photo.jpg"},
    {"id": "rps_rock_01", "name": "Rock (day 7 dataset)", "img": "img/rps_rock_01.jpg"},
    {"id": "rps_paper_07", "name": "Paper (day 7 dataset)", "img": "img/rps_paper_07.jpg"},
    {"id": "rps_scissors_01", "name": "Scissors (day 7 dataset)", "img": "img/rps_scissors_01.jpg"},
]

out_json = json.dumps(manifest, ensure_ascii=False, indent=1)
for banned in ("KAUST", "WISER", "junior researcher"):
    assert banned.lower() not in out_json.lower(), f"brand leak: {banned}"
(OUT / "studio.json").write_text(out_json, encoding="utf-8")
sizes = sum(f.stat().st_size for f in IMG.iterdir())
print(f"\nOK — studio.json ({len(out_json) // 1024} KB) + {len(list(IMG.iterdir()))} media files "
      f"({sizes // 1024} KB) in {OUT}")
