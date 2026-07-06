# ============================================================================
# AI Studio Internship - all-in-one GPU JupyterLab image
#
# Build:  docker build -t ai-studio:v3 E:\High_school_project_v3\docker
#         (add --build-arg PREFETCH=0 for a slim image without baked datasets)
#
# Run:    docker run --rm --gpus all --ipc=host ^
#           -p 127.0.0.1:8888:8888 -p 127.0.0.1:7860:7860 ^
#           -v "E:\High_school_project_v3:/workspace" ai-studio:v3
#
# Then open http://127.0.0.1:8888  (JupyterLab, no token needed)
# Gradio apps launched from notebooks appear at http://127.0.0.1:7860
# ============================================================================
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Deterministic torch/torchvision pair (the runtime base image is not
# guaranteed to bundle torchvision; this is a no-op for torch itself).
RUN pip install torch==2.12.1 torchvision==0.27.1 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt

# Bake datasets + model weights into the image, OUTSIDE /workspace so the
# volume mount never shadows them. Notebooks read COURSE_DATA_DIR first and
# fall back to ./data when it is not set (laptops / Colab).
ENV HF_HOME=/opt/hf-cache \
    TORCH_HOME=/opt/torch-cache \
    COURSE_DATA_DIR=/opt/course/data
ARG PREFETCH=1
COPY prefetch.py /tmp/prefetch.py
# If docker/cifar-10-python.tar.gz exists in the build context it is used
# directly (no network); otherwise prefetch.py downloads from a fast mirror.
# The tarball is removed after extraction so it never bloats a layer.
RUN --mount=type=bind,source=.,target=/ctx \
    if [ "$PREFETCH" = "1" ]; then \
        mkdir -p "$COURSE_DATA_DIR" && \
        { [ -f /ctx/cifar-10-python.tar.gz ] && \
          cp /ctx/cifar-10-python.tar.gz "$COURSE_DATA_DIR/" || true; } && \
        python /tmp/prefetch.py && \
        rm -f "$COURSE_DATA_DIR/cifar-10-python.tar.gz"; \
    fi

ENV GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /workspace
EXPOSE 8888 7860

# No token: the README run command binds ports to 127.0.0.1 only.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.root_dir=/workspace", \
     "--IdentityProvider.token=", "--ServerApp.password="]
