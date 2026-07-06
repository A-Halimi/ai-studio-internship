"""AI Studio - the app the interns assemble, one module per day.

Each file in modules/ named day*.py is a module with three things:

    TITLE    = "Digit Reader"            # tab name
    REQUIRES = ["digit_reader.pt"]       # files expected in ai_studio/models/
    def render(models_dir): ...          # builds the tab's UI (Gradio)

A module whose REQUIRES files are missing shows up as a locked tab -
finish that day's notebook (which exports the files) to unlock it.
"""
from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path

import gradio as gr

STUDIO_DIR = Path(__file__).resolve().parent
MODULES_DIR = STUDIO_DIR / "modules"
MODELS_DIR = STUDIO_DIR / "models"

HEADER = """
# 🧪 AI Studio
*One lab. Ten days. Every tab below is an AI capability you built yourself.*
"""


def load_modules() -> list[dict]:
    """Import every modules/day*.py, tolerating broken files."""
    entries: list[dict] = []
    for py in sorted(MODULES_DIR.glob("day*.py")):
        entry = {
            "name": py.stem,          # e.g. "day05_digit_reader"
            "title": py.stem,
            "requires": [],
            "render": None,
            "error": None,
        }
        try:
            spec = importlib.util.spec_from_file_location(
                f"ai_studio.modules.{py.stem}", py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            entry["title"] = getattr(mod, "TITLE", py.stem)
            entry["requires"] = list(getattr(mod, "REQUIRES", []))
            entry["render"] = getattr(mod, "render", None)
            if entry["render"] is None:
                entry["error"] = f"{py.name} has no render() function"
        except Exception:  # noqa: BLE001 - a broken module must not kill the app
            entry["error"] = traceback.format_exc()
        entries.append(entry)
    return entries


def missing_files(requires: list[str]) -> list[str]:
    return [f for f in requires if not (MODELS_DIR / f).exists()]


def build_demo() -> gr.Blocks:
    entries = load_modules()
    unlocked = sum(
        1 for e in entries
        if not e["error"] and not missing_files(e["requires"])
    )

    with gr.Blocks(title="AI Studio") as demo:
        gr.Markdown(HEADER)
        gr.Markdown(f"**{unlocked} / {len(entries)} modules unlocked**")
        with gr.Tabs():
            for e in entries:
                day = e["name"].split("_")[0]          # "day05"
                day_num = int(day.replace("day", ""))
                miss = missing_files(e["requires"])
                locked = bool(miss) or bool(e["error"])
                label = ("🔒 " if locked else "") + e["title"]
                with gr.Tab(label):
                    if e["error"]:
                        gr.Markdown(
                            "### ⚠️ This module crashed while loading\n"
                            "Show this to your supervisor:\n"
                            f"```\n{e['error']}\n```"
                        )
                    elif miss:
                        files = ", ".join(f"`{m}`" for m in miss)
                        gr.Markdown(
                            f"### 🔒 Locked\n"
                            f"Finish the **Day {day_num}** notebook to unlock "
                            f"**{e['title']}**.\n\n"
                            f"Its final *'Unlock your Studio module'* cell saves "
                            f"these missing files into `ai_studio/models/`: {files}"
                        )
                    else:
                        try:
                            e["render"](MODELS_DIR)
                        except Exception:  # noqa: BLE001
                            gr.Markdown(
                                "### ⚠️ Module failed to start\n"
                                f"```\n{traceback.format_exc()}\n```"
                            )
        gr.Markdown(
            "---\n*Under the hood, every one of these tabs is doing the same "
            "thing: **matrix multiplication**. Remember that in Week 3... 😉*"
        )
    return demo
