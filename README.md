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

### 🐳 Mode A: Docker (instructor workstation / any NVIDIA machine)

Everything pre-installed, datasets and models pre-baked, JupyterLab starts
automatically:

```powershell
# build once (~15 min; ~10.4 GB image)
docker build -t ai-studio:v3 E:\High_school_project_v3\docker

# run (JupyterLab: http://127.0.0.1:8888 — no token; Gradio apps: http://127.0.0.1:7860)
docker run --rm --gpus all --ipc=host `
  -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 `
  -v "E:\High_school_project_v3:/workspace" ai-studio:v3
```

Or `cd docker && docker compose up`. Notes: needs Docker Desktop with the
WSL2 backend and GPU support enabled; `--ipc=host` matters (DataLoader
workers); without a GPU drop `--gpus all` — everything falls back to CPU.

### ☁️ Mode B: Google Colab (students, zero install)

1. Push this repo to GitHub (see below).
2. Students open `notebooks/dayNN/<file>.ipynb` via
   `https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/...`
3. The first cell of every notebook detects Colab, clones the repo, and
   installs the few extra packages that day needs. Runtime → **T4 GPU**
   recommended on days 5–9.

### 💻 Mode C: laptop (GPU or CPU)

```bash
# NVIDIA GPU laptops — install the CUDA build of torch FIRST:
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# CPU-only laptops — just:
pip install -r requirements.txt

python scripts/download_data.py          # datasets (~250 MB; add --models for HF weights)
jupyter lab
```

Every notebook auto-detects the device; no GPU = slower, never broken.

---

## One-time setup for the instructor

1. **Set the GitHub URL** (needed only for Colab): push this repo, then
   search-replace `https://github.com/CHANGE-ME/ai-studio-internship` with
   your URL across `notebooks/` and `solutions/` (it lives in each setup cell).
2. **Print handouts**: PDFs are in `handouts/pdf/` (regenerate with
   `powershell -ExecutionPolicy Bypass -File scripts\build_pdfs.ps1`).
3. **Present slides**: open `slides/dayNN_*.html` in any browser —
   `F` = fullscreen, `←/→` = navigate, bottom strip = jump. Fully offline.
4. Read `INSTRUCTOR_GUIDE.md` — per-day runbook, pitfalls, and timings.

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

## Maintainer scripts

| Command | What it does |
|---|---|
| `python scripts/make_student_notebooks.py` | regenerate `notebooks/` from `solutions/` (strips `### BEGIN/END SOLUTION`, checks for leaks) |
| `python scripts/validate_notebooks.py` | nbformat-validate every notebook |
| `powershell -File scripts\build_pdfs.ps1` | handouts HTML → PDF (headless Chrome/Edge) |
| `python scripts/download_data.py [--models]` | laptop dataset/model fetch |
| in-container: `bash scripts/smoke_test.sh` | execute every solutions notebook (SMOKE_TEST=1 shrinks workloads) |

Full smoke test on this machine:

```powershell
docker run --rm --gpus all --ipc=host -e SMOKE_TEST=1 `
  -v "E:\High_school_project_v3:/workspace" ai-studio:v3 bash scripts/smoke_test.sh
```

## Troubleshooting

- **Slides render garbled characters** → your editor saved them as non-UTF-8;
  restore from git.
- **`docker: could not select device driver`** → enable GPU in Docker Desktop
  (Settings → Resources → WSL integration) and update the NVIDIA driver.
- **CIFAR-10 download is glacial** → the official server throttles; our code
  already uses a fast md5-verified mirror. In Docker it's pre-baked.
- **Gradio app not reachable from the container** → it must be the FIRST
  launched app (port 7860 is the one that's mapped); restart the kernel if a
  stray app holds the port.
- **A Studio tab shows 🔒** → that day's notebook hasn't exported its model
  files yet — the tab names the exact missing files.

---

*Built for the WISER summer internship. Weeks 3–4 continue with matrix
multiplication and HPC in Julia — the students will know exactly why.* 😉
