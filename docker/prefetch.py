"""Bake course datasets and model weights into the Docker image at build time.

Runs during `docker build` (skip with --build-arg PREFETCH=0). Everything is
cached OUTSIDE /workspace so the volume mount never shadows it:
  datasets      -> $COURSE_DATA_DIR (/opt/course/data)
  HF models     -> $HF_HOME         (/opt/hf-cache)
  torchvision   -> $TORCH_HOME      (/opt/torch-cache)
"""
import os

DATA_DIR = os.environ.get("COURSE_DATA_DIR", "/opt/course/data")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    from torchvision import datasets

    print(">>> MNIST (Day 5) ...", flush=True)
    datasets.MNIST(DATA_DIR, train=True, download=True)
    datasets.MNIST(DATA_DIR, train=False, download=True)

    print(">>> CIFAR-10 (Day 6) ...", flush=True)
    # The official cs.toronto.edu server is often extremely slow or drops
    # connections; try a fast md5-verified mirror first. If a valid tarball
    # was already copied into DATA_DIR (Docker build does this), no network
    # is used at all.
    CIFAR_MIRRORS = [
        "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ]
    last_err = None
    for url in CIFAR_MIRRORS:
        try:
            datasets.CIFAR10.url = url
            datasets.CIFAR10(DATA_DIR, train=True, download=True)
            datasets.CIFAR10(DATA_DIR, train=False, download=True)
            last_err = None
            break
        except Exception as err:  # noqa: BLE001
            print(f"    mirror failed ({url}): {err}", flush=True)
            last_err = err
    if last_err is not None:
        raise last_err

    from torchvision import models

    print(">>> MobileNetV3-Small + ResNet18 weights (Days 1 & 7) ...", flush=True)
    models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    from transformers import pipeline

    print(">>> Hugging Face models (Days 1 & 8) ...", flush=True)
    pipeline("text-generation", model="gpt2")
    pipeline("text-classification",
             model="distilbert-base-uncased-finetuned-sst-2-english")
    pipeline("fill-mask", model="distilbert-base-uncased")
    pipeline("zero-shot-classification",
             model="typeform/distilbert-base-uncased-mnli")

    print(">>> Prefetch complete.", flush=True)


if __name__ == "__main__":
    main()
