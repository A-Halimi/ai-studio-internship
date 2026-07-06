"""Run the AI Studio.

    python ai_studio/app.py

In the course Docker container it serves on http://127.0.0.1:7860
On Google Colab it prints a public share link.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_studio.studio import build_demo  # noqa: E402


def main() -> None:
    in_colab = "google.colab" in sys.modules
    demo = build_demo()
    demo.launch(share=in_colab)


if __name__ == "__main__":
    main()
