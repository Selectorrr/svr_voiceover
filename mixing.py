from glob import glob
from pathlib import Path

from pqdm.threads import pqdm
from pydub import AudioSegment

from modules.AudioProcessor import AudioProcessor

BASE_DIR = "workspace"


def _pad_to(seg: AudioSegment, dur_ms: int) -> AudioSegment:
    if len(seg) >= dur_ms:
        return seg
    # важно: frame_rate подхватываем от seg
    return seg + AudioSegment.silent(duration=dur_ms - len(seg), frame_rate=seg.frame_rate)


def mixing(meta: dict):
    dub_wave, dub_sr = AudioProcessor.load_audio(f"{src_dir}/{meta['src']}")
    dub_segment = AudioProcessor.to_segment(dub_wave, dub_sr)

    r_wave, r_sr = AudioProcessor.load_audio(f"{BASE_DIR}/resources/{meta['resource']}")
    resource_segment = AudioProcessor.to_segment(r_wave, r_sr)

    raw_audio = resource_segment - 9

    # Запоминаем "целевую" канальность (по resource)
    target_channels = raw_audio.channels

    # Выравниваем sample rate (до mono, чтобы меньше возни)
    if raw_audio.frame_rate != dub_segment.frame_rate:
        raw_audio = raw_audio.set_frame_rate(dub_segment.frame_rate)

    # ---- приведение к MONO ----
    if raw_audio.channels != 1:
        raw_mono = raw_audio.set_channels(1)      # N -> 1 (pydub умеет)
    else:
        raw_mono = raw_audio

    if dub_segment.channels != 1:
        dub_mono = dub_segment.set_channels(1)    # N -> 1 (pydub умеет)
    else:
        dub_mono = dub_segment

    # Выравнивание длительности (уже в mono)
    max_duration = max(len(raw_mono), len(dub_mono))
    raw_mono = _pad_to(raw_mono, max_duration)
    dub_mono = _pad_to(dub_mono, max_duration)

    # ---- сведение в MONO ----
    mixed_mono = raw_mono.overlay(dub_mono)

    # ---- восстановление каналов ----
    mixed_audio = mixed_mono.set_channels(target_channels) if target_channels != 1 else mixed_mono

    out_path = Path(f"{BASE_DIR}/vo/{meta['src']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wave_format = out_path.suffix[1:]
    codec = "libvorbis" if wave_format == "ogg" else None
    parameters = ["-qscale:a", "9"] if wave_format == "ogg" else None
    mixed_audio.export(out_path, format=wave_format, codec=codec, parameters=parameters)


src_dir = f"{BASE_DIR}/dub_aligned/"
if not Path(src_dir).exists():
    src_dir = f"{BASE_DIR}/dub_lipsync/"
    if not Path(src_dir).exists():
        src_dir = f"{BASE_DIR}/dub/"


if __name__ == "__main__":
    index = {}

    for resource_path in glob("**/*.*", root_dir=f"{BASE_DIR}/resources", recursive=True):
        resource_key = str(Path(resource_path).with_suffix(".key")).lower()
        meta = index.get(resource_key, {})
        meta["resource"] = resource_path
        index[resource_key] = meta

    for src_path in glob("**/*.*", root_dir=src_dir, recursive=True):
        src_key = str(Path(src_path).with_suffix(".key")).lower()
        meta = index.get(src_key, {})
        meta["src"] = src_path
        index[src_key] = meta

    for vo_path in glob("**/*.*", root_dir=f"{BASE_DIR}/vo", recursive=True):
        vo_key = str(Path(vo_path).with_suffix(".key")).lower()
        meta = index.get(vo_key, {})
        meta["vo"] = vo_path
        index[vo_key] = meta

    for key, value in list(index.items()):
        if "resource" not in value or "src" not in value or "vo" in value:
            del index[key]

    pqdm(index.values(), mixing, 24, smoothing=0)
