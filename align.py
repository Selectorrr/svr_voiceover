import argparse
import io
import math
import os
from glob import glob
from pathlib import Path

import librosa
import numpy
import soundfile
from pqdm.processes import pqdm
from pydub.silence import detect_nonsilent
from sympy.physics.quantum.matrixutils import to_numpy

from modules.AudioProcessor import AudioProcessor


def speedup(wave, sr, ratio, max_len, increment=0.05):
    segment = AudioProcessor.to_segment(wave, sr)
    ratio = min(2 - increment, ratio)
    ratio += increment
    while len(segment) > max_len * 1000:
        memory_buffer = io.BytesIO()
        segment.export(memory_buffer, format='wav', parameters=["-af", f"atempo={ratio}"])
        memory_buffer.seek(0)
        wave, sr = soundfile.read(memory_buffer)
        segment = AudioProcessor.to_segment(wave, sr)
        ratio = 1 + increment

    wave, sr = AudioProcessor.to_ndarray(segment)
    return wave


def get_sound_center(signal, sr, top_db=40):
    if signal.ndim > 1:
        mono = numpy.mean(signal, axis=tuple(range(1, signal.ndim)))
    else:
        mono = signal
    _, idx = trim(mono, sr)
    start, end = idx
    return start


def pad_or_trim_to_len(target_len, wave):
    if wave.shape[0] < target_len:
        pad = numpy.zeros((target_len - wave.shape[0],) + wave.shape[1:], dtype=wave.dtype)
        wave = numpy.concatenate([wave, pad], axis=0)
    else:
        wave = wave[:target_len]
    return wave


def audio_len(wav, sr):
    audio_length = len(wav) / sr
    return audio_length


def voice_len(y, sr, lower_bound=1):
    y = rtrim_audio(y, sr, lower_bound)
    r_len = audio_len(y, sr)
    return r_len


def rtrim_audio(y, sr, lower_bound=1):
    orig_len = audio_len(y, sr)
    if not lower_bound or orig_len <= lower_bound:
        return y
    if y.ndim > 1:
        mono = numpy.mean(y, axis=tuple(range(1, y.ndim)))
    else:
        mono = y
    _, idx = trim(mono, sr)
    _, end = idx
    y = y[:end]
    return y


def align_by_samples(wave, wave_sr, raw_wave, raw_sr, top_db=40, is_use_voice_len=False):
    if wave_sr != raw_sr:
        raw_wave, raw_sr = AudioProcessor.to_ndarray(AudioProcessor.to_segment(raw_wave, raw_sr).set_frame_rate(wave_sr))

    orig_len = raw_wave.shape[0]
    if is_use_voice_len:
        target_len = int(voice_len(raw_wave, raw_sr) * raw_sr)
    else:
        target_len = orig_len

    start_wave = get_sound_center(wave, wave_sr, top_db)
    start_raw = get_sound_center(raw_wave, raw_sr, top_db)

    desired_shift = start_raw - start_wave

    max_left_shift = start_wave
    max_right_shift = target_len - (wave.shape[0] - start_wave)

    # Ограничиваем shift, чтобы полезный сигнал полностью влез
    safe_shift = int(numpy.clip(desired_shift, -max_left_shift, max_right_shift))

    # Применяем сдвиг
    if safe_shift > 0:
        pad = numpy.zeros((safe_shift,) + wave.shape[1:], dtype=wave.dtype)
        wave = numpy.concatenate([pad, wave], axis=0)
    elif safe_shift < 0:
        wave = wave[-safe_shift:]

    # Дополняем справа до нужной длины
    if is_use_voice_len:
        wave = pad_or_trim_to_len(target_len, wave)
    # и в любом случае возвращаем длину orig_len
    wave = pad_or_trim_to_len(orig_len, wave)
    return wave


def trim(wave, sr):
    if wave.ndim > 1:
        wave, sr = to_numpy(AudioProcessor.to_segment(wave, sr).set_channels(1))
    return librosa.effects.trim(wave, top_db=40)


def main(wave, sr, raw_wave, raw_sr, is_use_voice_len=False):
    wave, _ = trim(wave, sr)
    wave_len = len(wave) / sr
    raw_len = len(raw_wave) / raw_sr
    rate = wave_len / raw_len
    if rate > 1:
        # ускорим, но не сильнее чем на 50 процентов
        wave = speedup(wave, sr, rate, raw_len)
    wave = align_by_samples(wave, sr, raw_wave, raw_sr, is_use_voice_len=is_use_voice_len)
    return wave


def worker(task):
    meta, is_use_voice_len = task

    in_wav = meta['src']
    wave, sr = soundfile.read(in_wav)

    raw_path = meta['resource']
    raw_wave, raw_sr = soundfile.read(raw_path)  # оставил как в исходнике

    result_wave = main(wave, sr, raw_wave, raw_sr, is_use_voice_len=is_use_voice_len)

    out_path = Path(f"workspace/dub_aligned/{in_wav[len(src_dir):]}").with_suffix('.wav')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(out_path, result_wave, sr, format='wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-voice-len",
        action="store_true",
        help="Использовать длину голосовой части вместо полной длины raw_wave"
    )
    args = parser.parse_args()
    use_voice_len_flag = args.use_voice_len  # по умолчанию False

    src_dir = 'workspace/dub_lipsync'
    index = {}
    for src_path in glob("**/*.*", root_dir=src_dir, recursive=True):
        src_key = str(Path(src_path).with_suffix('.key')).lower()
        meta = index.get(src_key, {})
        meta['src'] = f"{src_dir}/{src_path}"
        index[src_key] = meta

    for speedup_path in glob("**/*.*", root_dir='workspace/dub_aligned', recursive=True):
        speedup_key = str(Path(speedup_path).with_suffix('.key')).lower()
        meta = index.get(speedup_key, {})
        meta['speedup'] = f"workspace/dub_aligned/{speedup_path}"
        index[speedup_key] = meta

    for resource_path in glob("**/*.*", root_dir='workspace/resources', recursive=True):
        resource_key = str(Path(resource_path).with_suffix('.key')).lower()
        meta = index.get(resource_key, {})
        meta['resource'] = f"workspace/resources/{resource_path}"
        index[resource_key] = meta

    for key, value in list(index.items()):
        if 'src' not in value.keys() or 'speedup' in value.keys():
            del index[key]

    # прокидываем флаг в worker через задачи
    tasks = [(meta, use_voice_len_flag) for meta in index.values()]
    pqdm(tasks, worker, n_jobs=os.cpu_count(), desc="Speedup", smoothing=0)
