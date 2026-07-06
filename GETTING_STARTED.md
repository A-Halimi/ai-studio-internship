# Getting Started — Step by Step

*AI Studio Internship · WISER @ KAUST · July 19–30, 2026*
Repo: <https://github.com/A-Halimi/ai-studio-internship> · Image: `abdelghafour1/ai-studio`

---

## 🧑‍🏫 For the instructor (Abdelghafour)

### Once, before Day 1 (≈ 1 hour, mostly waiting)

1. **Get the materials & image ready** *(already done on the E:\ workstation)*
   ```powershell
   git clone https://github.com/A-Halimi/ai-studio-internship.git
   cd ai-studio-internship
   docker build -f docker/Dockerfile -t abdelghafour1/ai-studio:latest .
   docker login
   docker push abdelghafour1/ai-studio:latest
   ```
   Rebuild + re-push whenever you change materials (rebuilds are fast — the
   heavy layers are cached; students just `docker pull` again).

2. **Dry-run the classroom setup** on your machine:
   ```powershell
   docker run --rm --gpus all --ipc=host `
     -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 `
     -v "E:\ai-studio-test:/workspace" abdelghafour1/ai-studio:latest
   ```
   Open <http://127.0.0.1:8888> — you should see `notebooks/`, `slides/`,
   `ai_studio/` etc. appear in the (initially empty) `E:\ai-studio-test`.

3. **Smoke-test the answer key** (from the repo, which has `solutions/`):
   ```powershell
   docker run --rm --gpus all --ipc=host -e SMOKE_TEST=1 `
     -v "E:\High_school_project_v3:/workspace" abdelghafour1/ai-studio:latest `
     bash scripts/smoke_test.sh
   ```
   Expect: `11 passed, 0 failed`. Side effect: trained model files land in
   `ai_studio/models/` → run `python ai_studio/app.py` and enjoy the full
   7/7 Studio — **this is your Day-1 demo of the finished product.**

4. **Run one notebook in Colab** (needs any Google account):
   open the Day-5 link from the student table below, Runtime → T4 GPU,
   Run all. This validates the path your students will use.

5. **Print** `handouts/pdf/day01.pdf` + `day02.pdf` (rest as the week goes).

### Every course day

| Time | What you do |
|---|---|
| 08:55 | Open `slides/dayNN_*.html`, press `F` for fullscreen |
| 09:00 | Briefing (~30 min) — end on the missions slide, hand out the day's PDF |
| 09:30–15:30 | Students work in the notebook; circulate; lunch checkpoint per `INSTRUCTOR_GUIDE.md` |
| 15:30 | Show & tell — the 3 questions on the deck's last slide; students demo their newly unlocked Studio module |
| after | Skim tomorrow's runbook section in `INSTRUCTOR_GUIDE.md` (5 min) |

Solutions live in `solutions/dayNN/*_SOLUTIONS.ipynb` — your answer key;
reveal single blocks when someone is truly stuck, never the file.

---

## 🧑‍🎓 For students — pick ONE of the three ways

### Way 1 — Docker (best if you have a laptop with an NVIDIA GPU)

1. Install **Docker Desktop** (Windows: enable the WSL2 backend when asked).
2. Open a terminal (PowerShell on Windows) and pull the course image once
   (~7 GB — do this at home or on fast Wi-Fi):
   ```
   docker pull abdelghafour1/ai-studio:latest
   ```
3. Create an empty folder for your work, e.g. `my-ai-studio`, then run:
   ```powershell
   docker run --rm --gpus all --ipc=host `
     -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 `
     -v "$PWD\my-ai-studio:/workspace" abdelghafour1/ai-studio:latest
   ```
   *macOS/Linux:* same, but `-v "$PWD/my-ai-studio:/workspace"` and use `\`
   for line breaks. *No NVIDIA GPU?* Remove `--gpus all` — all notebooks
   fall back to CPU automatically.
4. Open **<http://127.0.0.1:8888>** in your browser → JupyterLab with all
   course materials. Your work is saved in `my-ai-studio` on YOUR disk.
5. Day-to-day: just repeat step 3 (same command, same folder).
   When your Studio app runs, it's at **<http://127.0.0.1:7860>**.

### Way 2 — Google Colab (zero install, free GPU in the browser)

1. You need a Google account. Open the day's notebook via its link:

   | Day | Notebook link |
   |---|---|
   | 1a | <https://colab.research.google.com/github/A-Halimi/ai-studio-internship/blob/main/notebooks/day01/day01a_data_detective.ipynb> |
   | 1b | …`/notebooks/day01/day01b_ai_wow_demos.ipynb` |
   | 2 | …`/notebooks/day02/day02_prediction_machine.ipynb` |
   | 3 | …`/notebooks/day03/day03_pattern_finder.ipynb` |
   | 4 | …`/notebooks/day04/day04_inside_the_brain.ipynb` |
   | 5 | …`/notebooks/day05/day05_digit_reader.ipynb` |
   | 6 | …`/notebooks/day06/day06_photo_detective.ipynb` |
   | 7 | …`/notebooks/day07/day07_rps_arena.ipynb` |
   | 8 | …`/notebooks/day08/day08_language_lab.ipynb` |
   | 9 | …`/notebooks/day09/day09_game_master.ipynb` |
   | 10 | …`/notebooks/day10/day10_demo_day.ipynb` |

   (every link starts with `https://colab.research.google.com/github/A-Halimi/ai-studio-internship/blob/main`)
2. **Runtime → Change runtime type → T4 GPU** (matters on days 5–9).
3. Run the first cell — it sets everything up (clones the course, installs
   that day's packages). Then work top to bottom.
4. ⚠️ Colab forgets files when the session ends. Download anything you want
   to keep (File → Download), especially your trained models on days 5–9,
   or save a copy of the notebook to your Drive (File → Save a copy).

### Way 3 — your own Python (no Docker, no Colab)

1. Install Python 3.11+ and git, then:
   ```bash
   git clone https://github.com/A-Halimi/ai-studio-internship.git
   cd ai-studio-internship
   # GPU laptop first (skip this line on CPU-only):
   pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   python scripts/download_data.py
   jupyter lab
   ```
2. Open `notebooks/day01/day01a_data_detective.ipynb` and go.

### Rules of thumb (all three ways)

- Work through the day's notebook top to bottom; fill every
  `YOUR CODE HERE`; the hints are right there in the comments.
- Finish the day's **🔓 Unlock** cell, then run `python ai_studio/app.py`
  (from the course folder) — your new module is live. Show it at show & tell.
- Stuck > 15 minutes → ask. Error messages are clues, not failures.

---

## ❓ FAQ

**Does the Docker image contain all the materials?**
Yes — since v3 it bakes the full *student* set: notebooks, slides, handouts
(+PDFs), the AI Studio app, sample data, and all datasets/model weights.
`solutions/` and pre-trained module models are deliberately NOT inside.

**Do Docker students need the GitHub repo?**
No. Pull + run is enough; materials appear on first start, and nothing in
the notebooks needs GitHub when running inside the container. The repo is
needed only for Colab (Way 2), the no-Docker laptop path (Way 3), and for
getting material updates without re-pulling the image.

**Is `-v` necessary?**
Not strictly — the container works without it. But without `-v` your edits
and trained models live only inside the container (and `--rm` deletes the
container at exit). Mount an empty folder with `-v` and everything persists
on your own disk. Short version: **always use `-v`, point it at an empty
folder, reuse the same folder every day.**
