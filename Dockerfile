# syntax=docker/dockerfile:1

ARG CUDA_IMAGE=nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# =========================
# builder
# =========================
FROM ${CUDA_IMAGE} AS builder
LABEL authors="SynthVoiceRu"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils python3.11-dev \
      build-essential gcc g++ pkg-config \
      git ffmpeg \
          libavcodec60 libavformat60 libavutil58 libavfilter9 libavdevice60 \
      libswresample4 libswscale7 \
      libgomp1 \
      wget tar \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install -U pip setuptools wheel

WORKDIR /workspace/SynthVoiceRu

# максимум кэшируем зависимости
COPY constraints.txt ./constraints.txt
COPY vendors/CosyVoice/requirements.txt vendors/CosyVoice/requirements.txt

# 1) Базовые pinned версии (всё под constraints)
# torch/torchaudio — из cu130
RUN python -m pip install -c constraints.txt \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.9.1+cu130 \
      torchaudio==2.9.1+cu130 \
      torchvision==0.24.1+cu130 \
      torchcodec==0.9.1+cu130

RUN python -m pip install -c constraints.txt \
      numpy==2.0.2 protobuf==6.33.3 onnx==1.20.1

# ORT CUDA 13 nightly (wheel без deps), deps ставим отдельно и тоже под constraints
RUN python -m pip install -c constraints.txt flatbuffers packaging sympy coloredlogs \
    && python -m pip install -c constraints.txt --no-deps --pre \
      --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
      onnxruntime-gpu==1.24.0.dev20260111001

# 2) пакеты
RUN python -m pip install -c constraints.txt \
      pyworld-prebuilt==0.3.5.post2 \
    && python -m pip install -c constraints.txt  \
      matcha-tts

# CosyVoice зависимости (тоже под constraints)
RUN python -m pip install -c constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r vendors/CosyVoice/requirements.txt

# vgmstream-cli
RUN wget -O /tmp/vgmstream-linux-cli.tar.gz \
      https://github.com/vgmstream/vgmstream-releases/releases/download/nightly/vgmstream-linux-cli.tar.gz \
    && tar -xvzf /tmp/vgmstream-linux-cli.tar.gz -C /tmp \
    && mv /tmp/vgmstream-cli /usr/local/bin/ \
    && chmod +x /usr/local/bin/vgmstream-cli \
    && rm -f /tmp/vgmstream-linux-cli.tar.gz


# =========================
# runtime
# =========================
FROM ${CUDA_IMAGE} AS runtime
LABEL authors="SynthVoiceRu"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/workspace/SynthVoiceRu/vendors/CosyVoice"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils \
      ffmpeg  \
      libpython3.11 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/vgmstream-cli /usr/local/bin/vgmstream-cli

WORKDIR /workspace/SynthVoiceRu

COPY constraints.txt ./constraints.txt
COPY requirements.txt ./requirements.txt
RUN python -m pip install -c constraints.txt -r requirements.txt

COPY utils/init_models.py /workspace/SynthVoiceRu/utils/init_models.py
RUN python /workspace/SynthVoiceRu/utils/init_models.py
COPY utils/init_cv3.py /workspace/SynthVoiceRu/utils/init_cv3.py
RUN python /workspace/SynthVoiceRu/utils/init_cv3.py

RUN python -m pip install -c constraints.txt \
      svr_tts==0.11.3

COPY . /workspace/SynthVoiceRu

ENTRYPOINT ["python", "entrypoint.py"]
