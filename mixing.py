import os
import subprocess
import tempfile
from functools import lru_cache
from glob import glob
from pathlib import Path

import soundfile
from pqdm.threads import pqdm
from pydub import AudioSegment

from modules.AudioProcessor import AudioProcessor

BASE_DIR = "workspace"


def _read_wem_in_memory(wem_file_path):
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_wav = temp_file.name  # Получаем имя временного файла

    try:
        # Конвертируем WEM в WAV через vgmstream
        process = subprocess.run(
            ["utils/vgmstream/vgmstream-cli.exe", wem_file_path, "-o", temp_wav],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if process.returncode != 0:
            raise RuntimeError(f"Ошибка конвертации: {process.stderr.decode()}")

        # Загружаем WAV в память
        audio, sr = soundfile.read(temp_wav)
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    return audio, sr


@lru_cache(maxsize=1000)
def load_audio(path):
    if Path(path).with_suffix('.wem').is_file():
        path = str(Path(path).with_suffix('.wem'))
    if path.endswith(".wem"):
        wave, sr = _read_wem_in_memory(path)
        segment = AudioProcessor.to_segment(wave, sr)
    elif path.endswith(".ogg"):
        segment = AudioSegment.from_ogg(path)
    elif path.endswith(".wav"):
        segment = AudioSegment.from_wav(path)
    else:
        raise ValueError('Unsupported file')
    channels = segment.channels
    if channels > 1:
        segment = segment.set_channels(1)
    return segment


def mixing(meta: dict):
    dub_segment = AudioSegment.from_file(f"{src_dir}/{meta['src']}")
    resource_segment = AudioSegment.from_file(f"{BASE_DIR}/resources/{meta['resource']}")
    raw_audio = resource_segment - 9
    if raw_audio.frame_rate != dub_segment.frame_rate:
        raw_audio = raw_audio.set_frame_rate(dub_segment.frame_rate)

    # Выравнивание длительности аудио
    max_duration = max(len(raw_audio), len(dub_segment))
    raw_audio = raw_audio + AudioSegment.silent(duration=max_duration - len(raw_audio))
    dub_segment = dub_segment + AudioSegment.silent(duration=max_duration - len(dub_segment))

    # Смешивание
    mixed_audio = raw_audio.overlay(dub_segment)

    out_path = f"workspace/vo/{meta['src']}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mixed_audio.export(out_path, format=Path(out_path).suffix[1:])


src_dir = f"{BASE_DIR}/dub_aligned/"
if not Path(src_dir).exists():
    src_dir = f"{BASE_DIR}/dub_lipsync/"
    if not Path(src_dir).exists():
        src_dir = f"{BASE_DIR}/dub/"

if __name__ == '__main__':
    index = {}
    for src_path in glob("**/*.*", root_dir=f"{BASE_DIR}/resources", recursive=True):
        src_stem = Path(src_path).stem.lower()
        meta = index.get(src_stem, {})
        meta['resource'] = src_path
        index[src_stem] = meta
    for src_path in glob("**/*.*", root_dir=src_dir, recursive=True):
        src_stem = Path(src_path).stem.lower()
        meta = index.get(src_stem, {})
        meta['src'] = src_path
        index[src_stem] = meta

    for vo_path in glob("**/*.*", root_dir='workspace/vo', recursive=True):
        vo_stem = Path(vo_path).stem.lower()
        meta = index.get(vo_stem, {})
        meta['vo'] = vo_path
        index[vo_stem] = meta

    for key, value in list(index.items()):
        if 'resource' not in value.keys() or 'src' not in value.keys():
            del index[key]
        if 'vo' in value.keys():
            del index[key]

    pqdm(index.values(), mixing, 24, smoothing=0)
