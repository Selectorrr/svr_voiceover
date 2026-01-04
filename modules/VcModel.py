import os
import subprocess
import tempfile
from pathlib import Path

import numpy
import soundfile
from chatterbox import ChatterboxVC
from huggingface_hub import snapshot_download

MODEL_SR = 24_000


class VcModel:
    def __init__(self, config, device):
        # скачает только папку s3gen в кеш HF (и будет переиспользовать кеш)
        root = Path(snapshot_download(
            repo_id="selectorrrr/svr-tts-large",
            repo_type="model",
            allow_patterns=["s3gen/**"],
            token=os.getenv("HF_TOKEN"),  # нужен только если репо приватное
        ))
        self.config = config

        self.vc = ChatterboxVC.from_local(str(root / "s3gen"), device)

    def __call__(self, full_wave, timbre_wave_24k, prosody_wave_24k) -> numpy.ndarray:
        timbre_wave_24k = timbre_wave_24k[: 5 * MODEL_SR]
        min_len = min(len(timbre_wave_24k), len(prosody_wave_24k))

        alpha = self.config['vc_default_alpha']  # 85% prosody, 15% timbre
        p_len = int(min_len * alpha)
        t_len = min_len - p_len

        timbre_wave = numpy.concatenate((
            timbre_wave_24k[:int(t_len / 2)],
            prosody_wave_24k[:p_len],
            timbre_wave_24k[int(len(timbre_wave_24k) - t_len / 2):],
        ))

        # отдельная временная папка на итерацию — без коллизий и проще чистить
        with tempfile.TemporaryDirectory(prefix="svr_") as tmpdir:
            tmpdir = Path(tmpdir)

            src_wav = tmpdir / "src.wav"
            tgt_wav = tmpdir / "target_24000.wav"
            out_wav = tmpdir / "out.wav"
            converted_wav = tmpdir / "converted.wav"

            def pad_to_mult(x: numpy.ndarray, mult: int):
                pad = (-len(x)) % mult
                return numpy.pad(x, (0, pad)) if pad else x

            MULT = int(0.04 * MODEL_SR)  # 960 при 24k

            src = numpy.asarray(full_wave, numpy.float32)
            tgt = numpy.asarray(timbre_wave, numpy.float32)

            tgt = pad_to_mult(tgt, MULT)

            soundfile.write(src_wav, src, samplerate=MODEL_SR)
            soundfile.write(tgt_wav, tgt, samplerate=MODEL_SR)

            wave = self.vc.generate(
                audio=str(src_wav),
                target_voice_path=str(tgt_wav),
            )
            wave = wave[0].detach().cpu().numpy()
            soundfile.write(str(converted_wav), wave, samplerate=MODEL_SR)
            self.mix_sibilants_ffmpeg(str(src_wav), str(converted_wav), str(out_wav))
            wave, _ = soundfile.read(str(out_wav))
        return wave

    @staticmethod
    def mix_sibilants_ffmpeg(
            source_wav: str,
            converted_wav: str,
            out_wav: str,
            fc: int = 8000,  # попробуй 7500..9500
            hi_gain_db: float = 6,  # попробуй 0..10
            sr: int = 24000,
    ):
        source_wav = str(source_wav)
        converted_wav = str(converted_wav)
        out_wav = str(out_wav)

        filt = (
            f"[0:a]aresample={sr},highpass=f={fc},treble=g={hi_gain_db}:f={fc}[s_hi];"
            f"[1:a]aresample={sr},lowpass=f={fc}[c_lo];"
            f"[c_lo][s_hi]amix=inputs=2:weights='1 1':normalize=0,"
            f"alimiter=limit=0.98[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", source_wav,
            "-i", converted_wav,
            "-filter_complex", filt,
            "-map", "[out]",
            "-ac", "1",  # чтобы не было сюрпризов со стерео
            "-ar", str(sr),
            out_wav,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed:\n{e.stderr}\ncmd={' '.join(cmd)}") from e
