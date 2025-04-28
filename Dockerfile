# === Этап сборки (builder) ===
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder
LABEL authors="SynthVoiceRu"

# Устанавливаем необходимые утилиты и Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    git \
    ffmpeg \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip (если ещё не установлен)
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Создаем виртуальное окружение и активируем его
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем зависимости и библиотеку (включая свою) через pip
RUN pip install --no-cache-dir svr_tts==0.4 \
    soundfile librosa pydub pyloudnorm GPUtil pqdm

# Скачиваем и устанавливаем vgmstream-cli
RUN wget https://github.com/vgmstream/vgmstream-releases/releases/download/nightly/vgmstream-linux-cli.tar.gz \
    && tar -xvzf vgmstream-linux-cli.tar.gz \
    && mv vgmstream-cli /usr/local/bin/ \
    && rm vgmstream-linux-cli.tar.gz

# === Этап выполнения (runtime) ===
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
LABEL authors="SynthVoiceRu"

# Устанавливаем runtime-утилиты и Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Обновляем LD_LIBRARY_PATH, чтобы искать библиотеки в каталоге CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Создаем символическую ссылку, если libcudnn.so.9 не найдена
RUN if [ ! -f /usr/local/cuda/lib64/libcudnn.so.9 ]; then \
      ln -s /usr/local/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.9; \
    fi


# Копируем виртуальное окружение из этапа сборки
COPY --from=builder /opt/venv /opt/venv
# Копируем vgmstream-cli
COPY --from=builder /usr/local/bin/vgmstream-cli /usr/local/bin/vgmstream-cli
ENV PATH="/opt/venv/bin:$PATH"

COPY utils/init_models.py /workspace/SynthVoiceRu/utils/init_models.py
RUN python3.11 /workspace/SynthVoiceRu/utils/init_models.py

# Копируем исходный код проекта
WORKDIR /workspace/SynthVoiceRu
COPY . /workspace/SynthVoiceRu

ENTRYPOINT ["python3.11", "entrypoint.py"]
