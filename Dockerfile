# === Этап сборки (builder) ===
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS builder
LABEL authors="SynthVoiceRu"

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 из deadsnakes + утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    git \
    ffmpeg \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip tooling
RUN pip install --no-cache-dir -U pip setuptools wheel

# зависимости проекта

RUN pip install --no-cache-dir \
    svr_tts==0.11.1 \
    soundfile librosa pydub pyloudnorm GPUtil pqdm \
    onnx-asr[audio] audalign

RUN pip install --no-cache-dir chatterbox-tts

#обновим pytorch
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

#обновим onnx
RUN pip uninstall -y onnxruntime onnxruntime-gpu || true
RUN pip install -U flatbuffers numpy packaging protobuf sympy coloredlogs
RUN pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu --no-deps

#решим конфликты после обновления
RUN pip install --no-cache-dir --force-reinstall \
    "numpy>=2.0,<2.4" \
    "scipy>=1.13,<2" \
    "numba<0.61" \
    "llvmlite<0.44" \
    "matplotlib>=3.10.8" \
    "pyloudnorm"

RUN pip install --no-cache-dir -U pyloudnorm

# vgmstream-cli
RUN wget -O /tmp/vgmstream-linux-cli.tar.gz \
      https://github.com/vgmstream/vgmstream-releases/releases/download/nightly/vgmstream-linux-cli.tar.gz \
    && tar -xvzf /tmp/vgmstream-linux-cli.tar.gz -C /tmp \
    && mv /tmp/vgmstream-cli /usr/local/bin/ \
    && chmod +x /usr/local/bin/vgmstream-cli \
    && rm -f /tmp/vgmstream-linux-cli.tar.gz


# === Этап выполнения (runtime) ===
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS runtime
LABEL authors="SynthVoiceRu"

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CUDA libs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


RUN if [ ! -f /usr/local/cuda/lib64/libcudnn.so.9 ]; then \
      ln -s /usr/local/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.9; \
    fi

# venv + vgmstream
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/vgmstream-cli /usr/local/bin/vgmstream-cli
ENV PATH="/opt/venv/bin:$PATH"

# init models (скрипт должен существовать в контексте сборки)
COPY utils/init_models.py /workspace/SynthVoiceRu/utils/init_models.py
RUN python3.11 /workspace/SynthVoiceRu/utils/init_models.py

# код проекта
WORKDIR /workspace/SynthVoiceRu
COPY . /workspace/SynthVoiceRu

ENTRYPOINT ["python3.11", "entrypoint.py"]
