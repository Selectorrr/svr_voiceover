import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy
import soundfile
import torch
from chatterbox import ChatterboxVC
from huggingface_hub import snapshot_download

MODEL_SR = 24_000
SPEAKER_SR = 16_000


class VcModel:
    def __init__(self, config, device):
        root = Path(snapshot_download(
            repo_id="selectorrrr/svr-tts-large",
            repo_type="model",
            allow_patterns=["s3gen/**"],
            token=os.getenv("HF_TOKEN"),
        ))
        self.config = config
        self.device = device
        self.vc = ChatterboxVC.from_local(str(root / "s3gen"), device)

        self.cache_dir = Path("workspace/voices/timbre_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _voice_key_from_timbre(timbre_wave_24k: numpy.ndarray, sr: int, sec: float = 10.0) -> str:
        n = int(sr * sec)
        x = numpy.asarray(timbre_wave_24k[:n], dtype=numpy.float32)
        h = hashlib.blake2b(digest_size=16)
        h.update(str(sr).encode("utf-8"))
        h.update(x.tobytes())
        return h.hexdigest()

    def _xvec_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def _load_xvec_cpu(self, key: str) -> torch.Tensor | None:
        p = self._xvec_cache_path(key)
        if not p.exists():
            return None
        try:
            t = torch.load(p, map_location="cpu")
            if torch.is_tensor(t):
                return t
        except Exception:
            pass
        return None

    def _save_xvec_cpu(self, key: str, xvec_cpu: torch.Tensor):
        p = self._xvec_cache_path(key)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(xvec_cpu, tmp)
        tmp.replace(p)

    def _compute_xvec_cpu(self, timbre_wave_24k: numpy.ndarray) -> torch.Tensor:
        # x-vector считается по 16k
        wav16 = librosa.resample(
            numpy.asarray(timbre_wave_24k, dtype=numpy.float32),
            orig_sr=MODEL_SR,
            target_sr=SPEAKER_SR,
        )
        if wav16.size == 0:
            raise ValueError("timbre_wave_24k is empty")

        wav16_t = torch.from_numpy(wav16).unsqueeze(0)  # (1, L)

        dtype = getattr(self.vc.s3gen, "dtype", torch.float32)
        with torch.inference_mode():
            xvec = self.vc.s3gen.speaker_encoder.inference(
                wav16_t.to(device=self.device, dtype=dtype)
            )

        # сохраняем на CPU (можно float16 чтобы меньше места)
        xvec_cpu = xvec.detach().to("cpu")
        if xvec_cpu.dtype == torch.float32:
            xvec_cpu = xvec_cpu.to(torch.float16)

        return xvec_cpu

    def __call__(self, full_wave, timbre_wave_24k_orig, prosody_wave_24k) -> numpy.ndarray:
        timbre_wave_24k = timbre_wave_24k_orig[: 10 * MODEL_SR]

        alpha = float(self.config['vc_default_alpha'])
        min_target_sec = float(self.config['min_target_sec'])
        alpha = max(0.0, min(1.0, alpha))

        target_len = int(round(min_target_sec * MODEL_SR))
        if target_len < 1:
            target_len = 1

        p_want = int(round(target_len * alpha))
        p_len = min(p_want, len(prosody_wave_24k))

        t_len = max(0, target_len - p_len)
        left_t = t_len // 2
        right_t = t_len - left_t

        left_t = min(left_t, len(timbre_wave_24k))
        right_t = min(right_t, len(timbre_wave_24k))

        mixed = numpy.concatenate((
            timbre_wave_24k[:left_t],
            prosody_wave_24k[:p_len],
            timbre_wave_24k[len(timbre_wave_24k) - right_t:],
        ))

        if target_len > len(mixed) > 0:
            reps = int(numpy.ceil(target_len / len(mixed)))
            mixed = numpy.tile(mixed, reps)[:target_len]

        def pad_to_mult(x: numpy.ndarray, mult: int):
            pad = (-len(x)) % mult
            return numpy.pad(x, (0, pad)) if pad else x

        MULT = int(0.04 * MODEL_SR)  # 960 при 24k

        src = numpy.asarray(full_wave, numpy.float32)
        tgt = numpy.asarray(mixed, numpy.float32)
        tgt = pad_to_mult(tgt, MULT)

        # --- x-vector cache (disk only) ---
        key = self._voice_key_from_timbre(timbre_wave_24k, MODEL_SR, sec=10.0)
        xvec_cpu = self._load_xvec_cpu(key)
        if xvec_cpu is None:
            xvec_cpu = self._compute_xvec_cpu(timbre_wave_24k)
            self._save_xvec_cpu(key, xvec_cpu)

        with tempfile.TemporaryDirectory(prefix="svr_") as tmpdir:
            tmpdir = Path(tmpdir)

            src_wav = tmpdir / "src.wav"
            tgt_wav = tmpdir / "target_24000.wav"
            out_wav = tmpdir / "out.wav"
            converted_wav = tmpdir / "converted.wav"

            soundfile.write(src_wav, src, samplerate=MODEL_SR)
            soundfile.write(tgt_wav, tgt, samplerate=MODEL_SR)

            # Подменяем speaker_encoder.inference => возвращаем кэшированный x-vector
            speaker_encoder = self.vc.s3gen.speaker_encoder
            orig_inference = speaker_encoder.inference

            dtype = getattr(self.vc.s3gen, "dtype", torch.float32)
            xvec_dev = xvec_cpu.to(device=self.device, dtype=dtype, non_blocking=True)

            def cached_inference(wav16: torch.Tensor):
                b = int(wav16.shape[0]) if wav16 is not None and wav16.ndim >= 1 else 1
                if xvec_dev.shape[0] == b:
                    return xvec_dev
                return xvec_dev.expand(b, -1)

            speaker_encoder.inference = cached_inference
            try:
                wave = self.vc.generate(audio=str(src_wav), target_voice_path=str(tgt_wav))
            finally:
                speaker_encoder.inference = orig_inference

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
            fc: int = 8000,
            hi_gain_db: float = 6,
            sr: int = 24000,
    ):
        filt = (
            f"[0:a]aresample={sr},highpass=f={fc},treble=g={hi_gain_db}:f={fc},deesser[s_hi];"
            f"[1:a]aresample={sr},lowpass=f={fc}[c_lo];"
            f"[c_lo][s_hi]amix=inputs=2:weights='1 1':normalize=0,"
            f"alimiter=limit=0.98[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(source_wav),
            "-i", str(converted_wav),
            "-filter_complex", filt,
            "-map", "[out]",
            "-ac", "1",
            "-ar", str(sr),
            str(out_wav),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed:\n{e.stderr}\ncmd={' '.join(cmd)}") from e
