"""Download the course datasets into ./data (laptop setup).

    python scripts/download_data.py            # datasets only (~250 MB)
    python scripts/download_data.py --models   # + pretrained model weights (~1.7 GB)

Docker users can skip this: the image already contains everything.
Colab users can skip this too: each notebook downloads what it needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true",
                        help="also pre-download pretrained model weights")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    from torchvision import datasets

    print(">>> MNIST (Day 5) ...")
    datasets.MNIST(DATA_DIR, train=True, download=True)
    datasets.MNIST(DATA_DIR, train=False, download=True)

    print(">>> CIFAR-10 (Day 6) ...")
    # Fast md5-verified mirror first; the official server is often slow.
    for url in (
        "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ):
        try:
            datasets.CIFAR10.url = url
            datasets.CIFAR10(DATA_DIR, train=True, download=True)
            datasets.CIFAR10(DATA_DIR, train=False, download=True)
            break
        except Exception as err:  # noqa: BLE001
            print(f"    mirror failed ({url}): {err}")
    else:
        raise RuntimeError("Could not download CIFAR-10 from any mirror.")

    if args.models:
        from torchvision import models

        print(">>> MobileNetV3-Small + ResNet18 weights ...")
        models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        from transformers import pipeline

        print(">>> Hugging Face models (Days 1 & 8) ...")
        pipeline("text-generation", model="gpt2")
        pipeline("text-classification",
                 model="distilbert-base-uncased-finetuned-sst-2-english")
        pipeline("fill-mask", model="distilbert-base-uncased")
        pipeline("zero-shot-classification",
                 model="typeform/distilbert-base-uncased-mnli")

    print(">>> Done. Data in", DATA_DIR)


if __name__ == "__main__":
    main()
