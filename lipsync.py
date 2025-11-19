import os
import tempfile
from glob import glob
from pathlib import Path

import audalign as ad
import soundfile as sf
from pqdm.processes import pqdm
from pydub import AudioSegment


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
        AudioSegment.from_file(ref_path, format=f"{Path(ref_path).suffix[1:]}").export(ref_wav, format="wav")
        AudioSegment.from_file(src_path, format=f"{Path(src_path).suffix[1:]}").export(src_wav, format="wav")

        sr_src = sf.info(src_path).samplerate

        cand = align_by_ref(ref_wav, src_wav, sr_src, td)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        AudioSegment.from_wav(cand).export(out_path, format=Path(out_path).suffix[1:])

        # sanity-check
        sr_out = sf.info(out_path).samplerate
        if sr_out != sr_src:
            raise RuntimeError(f"SR changed! src={sr_src}, out={sr_out}")

src_dir = "workspace/dub"
def main():
    global src_dir
    index = {}
    for src_path in glob("**/*.*", root_dir=src_dir, recursive=True):
        src_stem = Path(src_path).stem.lower()
        meta = index.get(src_stem, {})
        meta['src'] = src_path
        index[src_stem] = meta

    for lipsync_path in glob("**/*.*", root_dir='workspace/dub_lipsync', recursive=True):
        lipsync_stem = Path(lipsync_path).stem.lower()
        meta = index.get(lipsync_stem, {})
        meta['lipsync'] = lipsync_path
        index[lipsync_stem] = meta

    for resource_path in glob("**/*.*", root_dir='workspace/resources', recursive=True):
        resource_stem = Path(resource_path).stem.lower()
        meta = index.get(resource_stem, {})
        meta['resource'] = resource_path
        index[resource_stem] = meta

    for key, value in list(index.items()):
        if 'resource' not in value.keys() or 'src' not in value.keys():
            del index[key]
        if 'lipsync' in value.keys():
            del index[key]

    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    pqdm(index.values(), _worker, n_jobs=n_jobs, desc="Aligning")


if __name__ == "__main__":
    main()
