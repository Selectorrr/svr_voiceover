import os
import tempfile
from glob import glob
from pathlib import Path

import audalign as ad
import soundfile as sf
from pqdm.processes import pqdm
from pydub import AudioSegment

from modules.AudioProcessor import AudioProcessor


def align_by_ref(ref_wav: str, src_wav: str, sr_src: int, td):
    rec = ad.CorrelationRecognizer()
    rec.config.multiprocessing = False
    rec.config.num_processors = 1
    rec.config.sample_rate = sr_src

    base = ad.align_files(ref_wav, src_wav, recognizer=rec)
    fine = ad.fine_align(base, recognizer=rec)

    ad.write_shifts_from_results(
        fine,
        td,
        read_from_dir=str(Path(src_wav).parent),
        write_extension="wav",
        unprocessed=True
    )
    return td / Path(src_wav).name


# пример запуска
def _worker(meta: dict):
    ref_path = f"workspace/resources/{meta['resource']}"
    src_path = f"{src_dir}/{meta['src']}"
    out_path = f"workspace/dub_lipsync/{meta['src']}"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref_wav = os.path.join(td, "ref.wav")
        src_wav = os.path.join(td, "src.wav")
        dub_wave, dub_sr = AudioProcessor.load_audio(ref_path)
        ref_seg = AudioProcessor.to_segment(dub_wave, dub_sr)
        ref_seg.export(ref_wav, format="wav")
        AudioSegment.from_file(src_path, format=f"{Path(src_path).suffix[1:]}").export(src_wav, format="wav")

        sr_src = sf.info(src_path).samplerate

        cand = align_by_ref(ref_wav, src_wav, sr_src, td)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wave_format = out_path.suffix[1:]
        codec = "libvorbis" if wave_format == 'ogg' else None
        parameters = ["-qscale:a", "9"] if wave_format == 'ogg' else None
        AudioSegment.from_wav(cand).export(out_path, format=wave_format, codec=codec, parameters=parameters)


src_dir = "workspace/dub"
def main():
    global src_dir
    index = {}
    for src_path in glob("**/*.*", root_dir=src_dir, recursive=True):
        src_path_index = str(Path(src_path).with_suffix('.key')).lower()
        meta = index.get(src_path_index, {})
        meta['src'] = src_path
        index[src_path_index] = meta

    for lipsync_path in glob("**/*.*", root_dir='workspace/dub_lipsync', recursive=True):
        lipsync_index = str(Path(lipsync_path).with_suffix('.key')).lower()
        meta = index.get(lipsync_index, {})
        meta['lipsync'] = lipsync_path
        index[lipsync_index] = meta

    for resource_path in glob("**/*.*", root_dir='workspace/resources', recursive=True):
        resource_index = str(Path(resource_path).with_suffix('.key')).lower()
        meta = index.get(resource_index, {})
        meta['resource'] = resource_path
        index[resource_index] = meta

    for key, value in list(index.items()):
        if 'resource' not in value.keys() or 'src' not in value.keys() or 'lipsync' in value.keys():
            del index[key]

    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    pqdm(index.values(), _worker, n_jobs=n_jobs, desc="Aligning", smoothing=0)


if __name__ == "__main__":
    main()
