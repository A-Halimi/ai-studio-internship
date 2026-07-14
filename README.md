# 🧪 AI Studio — a 2-Week High-School AI Internship

**WISER Summer Internship · KAUST · July 19–30, 2026 (Sun–Thu)**

Students join *the Lab* as junior AI researchers. Each day they master one
pillar of AI and unlock one module of their personal **AI Studio** — a growing
web app. Day 10 is **Demo Day**: they assemble all seven modules, present
live, and run the *Grand Benchmark* that reveals the single operation
underneath everything — **matrix multiplication** — the perfect handoff to
Weeks 3–4 (HPC with Julia).

| Day | Date | Topic | Studio module unlocked |
|----:|------|-------|------------------------|
| 1 | Sun Jul 19 | Welcome to the Lab — data literacy + AI wow demos | — |
| 2 | Mon Jul 20 | Supervised learning (k-NN, trees) | 🐧 Prediction Machine |
| 3 | Tue Jul 21 | Unsupervised learning (k-means, PCA, t-SNE) | 🎨 Pattern Finder |
| 4 | Wed Jul 22 | Neural network from scratch (NumPy) | — (the engine day) |
| 5 | Thu Jul 23 | PyTorch + GPU, MNIST | ✍️ Digit Reader |
| 6 | Sun Jul 26 | CNNs, CIFAR-10 | 🕵️ Photo Detective |
| 7 | Mon Jul 27 | Transfer learning (webcam rock-paper-scissors) | ✂️ RPS Arena |
| 8 | Tue Jul 28 | Transformers & LLMs | 💬 Language Lab |
| 9 | Wed Jul 29 | Reinforcement learning | 🎮 Game Master |
| 10 | Thu Jul 30 | **Demo Day** + the road to HPC | 🚀 everything |

Daily rhythm: **morning briefing** (interactive HTML deck in `slides/`) →
**lab time** (notebook in `notebooks/dayNN/`) → **evening show & tell**
(questions are on the last slide + handout).

---

## Quick start — pick your mode

> Repo: <https://github.com/A-Halimi/ai-studio-internship> ·
> Image: [`abdelghafour1/ai-studio`](https://hub.docker.com/r/abdelghafour1/ai-studio)

### 🐳 Mode A: Docker (recommended — one pull, everything included)

The image contains the **full environment AND all course materials**
(notebooks, slides, handouts, the AI Studio app, datasets, model weights).
On first start it copies the materials into `/workspace`.

**Set up once** (creates a persistent, named container — pick your OS):

*Windows · PowerShell*
```powershell
docker pull abdelghafour1/ai-studio:latest
docker run -d --name ai-studio --gpus all --ipc=host `
  -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 `
  -v "${PWD}\my-ai-studio:/workspace" abdelghafour1/ai-studio:latest
```

*Windows · Command Prompt (cmd)*
```bat
docker pull abdelghafour1/ai-studio:latest
docker run -d --name ai-studio --gpus all --ipc=host ^
  -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 ^
  -v "%cd%\my-ai-studio:/workspace" abdelghafour1/ai-studio:latest
```

*macOS / Linux*
```bash
docker pull abdelghafour1/ai-studio:latest
docker run -d --name ai-studio --gpus all --ipc=host \
  -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 \
  -v "$PWD/my-ai-studio:/workspace" abdelghafour1/ai-studio:latest
```

Then open **<http://127.0.0.1:8888>** (JupyterLab, no token). Gradio apps:
**<http://127.0.0.1:7860>**.

**Alternative — host networking (no `-p`; every container port reachable).**
Swap the two `-p …` flags for `--network host`. Useful when a notebook opens a
second app (e.g. Gradio falls back to 7861 when 7860 is busy): with host
networking *every* port the container opens is reachable, no mapping needed.

*Linux:*
```bash
docker run -d --name ai-studio --gpus all --ipc=host --network host \
  -v "$PWD/my-ai-studio:/workspace" abdelghafour1/ai-studio:latest
```

*Windows (PowerShell) / macOS — Docker Desktop:* first enable
**Settings → Resources → Network → "Enable host networking"** (Docker Desktop
4.34+), then drop the `-p` lines and add `--network host`:
```powershell
docker run -d --name ai-studio --gpus all --ipc=host --network host `
  -v "${PWD}\my-ai-studio:/workspace" abdelghafour1/ai-studio:latest
```

Open the same **<http://127.0.0.1:8888>** / **<http://127.0.0.1:7860>**.

> ⚠️ **Security trade-off:** JupyterLab in the image runs with **no token or
> password**. The `-p 127.0.0.1:…` commands above keep it bound to your machine
> only; `--network host` exposes it on **every** network interface, so anyone on
> the same Wi-Fi/LAN can open it and run code. Use host networking only on a
> trusted network or a single-user machine — the published-port commands stay
> the safer classroom default.

**Every day after** — no long command needed:
```
docker start ai-studio      # continue where you left off
docker stop  ai-studio      # done for the day (or just shut the laptop)
```

Notes:
- **No NVIDIA GPU** (all Macs, many laptops): remove `--gpus all` —
  everything falls back to CPU. Apple-Silicon Macs run the image under
  emulation (works, but slow) — Colab is usually the better choice there.
- **`-v` (the folder mount)** is *optional but strongly recommended*: the
  materials are copied into `my-ai-studio` on first start and **all your
  work persists on your own disk** — even if the container is deleted.
  Reuse the same folder every time.
- **When the instructor ships an update**:
  `docker stop ai-studio && docker rm ai-studio && docker pull abdelghafour1/ai-studio:latest`
  then run the setup command again — your work is safe in `my-ai-studio`
  (already-present files are kept; delete a notebook to get its fresh copy).
- No git and no GitHub account needed in this mode.

### ☁️ Mode B: Google Colab (zero install, free GPU)

Open any student notebook directly, e.g. Day 1:

```
https://colab.research.google.com/github/A-Halimi/ai-studio-internship/blob/main/notebooks/day01/day01a_data_detective.ipynb
```

(Same pattern for every `notebooks/dayNN/<file>.ipynb`.) The first cell
detects Colab, clones this repo, and installs the few extra packages that
day needs. Runtime → **T4 GPU** recommended on days 5–9.

### 💻 Mode C: laptop, no Docker (GPU or CPU)

```bash
git clone https://github.com/A-Halimi/ai-studio-internship.git
cd ai-studio-internship

# NVIDIA GPU laptops — install the CUDA build of torch FIRST:
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
# CPU-only laptops — skip the first line, just:  pip install -r requirements.txt

python scripts/download_data.py          # datasets (~250 MB; add --models for HF weights)
jupyter lab
```

Every notebook auto-detects the device; no GPU = slower, never broken.

---


## Repository layout

```
slides/          10 interactive HTML briefings (self-contained, offline)
notebooks/       student notebooks (GENERATED — do not edit by hand)
solutions/       instructor masters with full solutions  ⚠️ don't share
handouts/        printable daily briefs (html sources + pdf)
ai_studio/       the Studio app: python ai_studio/app.py
  modules/         7 modules, one per day (auto-lock until models exist)
  models/          where student notebooks export trained models
scripts/         build & QA tooling (see below)
docker/          GPU JupyterLab image (Dockerfile, compose, prefetch)
data_samples/    small datasets shipped with the repo (penguins, RPS photos)
data/            big datasets, downloaded on demand (gitignored)
```



*Built for the WISER summer internship. Weeks 3–4 continue with matrix
multiplication and HPC in Julia — the students will know exactly why.* 😉
