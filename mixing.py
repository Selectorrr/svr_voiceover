from glob import glob
from pathlib import Path

from pqdm.threads import pqdm
from pydub import AudioSegment

BASE_DIR = "workspace"

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

    mixed_audio.export(out_path, format='wav')

src_dir = f"{BASE_DIR}/dub_aligned/"
if not Path(src_dir).exists():
    src_dir = f"{BASE_DIR}/dub_lipsync/"
    if not Path(src_dir).exists():
        src_dir = f"{BASE_DIR}/dub/"

if __name__ == '__main__':
    index = {}
    for resource_path in glob("**/*.*", root_dir=f"{BASE_DIR}/resources", recursive=True):
        resource_key = str(Path(resource_path).with_suffix('.key')).lower()
        meta = index.get(resource_key, {})
        meta['resource'] = resource_path
        index[resource_key] = meta
    for src_path in glob("**/*.*", root_dir=src_dir, recursive=True):
        src_key = str(Path(src_path).with_suffix('.key')).lower()
        meta = index.get(src_key, {})
        meta['src'] = src_path
        index[src_key] = meta

    for vo_path in glob("**/*.*", root_dir='workspace/vo', recursive=True):
        vo_key = str(Path(vo_path).with_suffix('.key')).lower()
        meta = index.get(vo_key, {})
        meta['vo'] = vo_path
        index[vo_key] = meta

    for key, value in list(index.items()):
        if 'resource' not in value.keys() or 'src' not in value.keys() or 'vo' in value.keys():
            del index[key]

    pqdm(index.values(), mixing, 24, smoothing=0)
