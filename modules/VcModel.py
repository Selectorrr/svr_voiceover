import os
import subprocess
import tempfile
from pathlib import Path
from collections import OrderedDict
import hashlib

import numpy
import soundfile
import torch
from chatterbox import ChatterboxVC
from huggingface_hub import snapshot_download

MODEL_SR = 24_000


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

        # CPU LRU cache: voice_key -> ref_dict_cpu (используем ТОЛЬКО при alpha==0)
        self._voice_cache = OrderedDict()
        self._voice_cache_max = int(self.config.get("vc_voice_cache_max", 200))

    @staticmethod
    def _voice_key_from_timbre(timbre_wave_24k: numpy.ndarray, sr: int, sec: float = 10.0) -> str:
        n = int(sr * sec)
        x = numpy.asarray(timbre_wave_24k[:n], dtype=numpy.float32)
        h = hashlib.blake2b(digest_size=16)
        h.update(str(sr).encode("utf-8"))
        h.update(x.tobytes())
        return h.hexdigest()

    def _cache_get(self, key: str):
        v = self._voice_cache.get(key)
        if v is None:
            return None
        self._voice_cache.move_to_end(key)
        return v

    def _cache_put(self, key: str, ref_cpu: dict):
        self._voice_cache[key] = ref_cpu
        self._voice_cache.move_to_end(key)
        while len(self._voice_cache) > self._voice_cache_max:
            self._voice_cache.popitem(last=False)

    def _set_vc_ref_from_cpu(self, ref_cpu: dict):
        # каждый раз переносим на device, кэш остаётся на CPU
        self.vc.ref_dict = {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in ref_cpu.items()
        }

    def __call__(self, full_wave, timbre_wave_24k, prosody_wave_24k) -> numpy.ndarray:
        timbre_wave_24k = timbre_wave_24k[: 5 * MODEL_SR]

        alpha = float(self.config['vc_default_alpha'])
        min_target_sec = float(self.config['min_target_sec'])
        alpha = max(0.0, min(1.0, alpha))
        eps = 1e-6

        def pad_to_mult(x: numpy.ndarray, mult: int):
            pad = (-len(x)) % mult
            return numpy.pad(x, (0, pad)) if pad else x

        MULT = int(0.04 * MODEL_SR)  # 960 при 24k

        src = numpy.asarray(full_wave, numpy.float32)

        with tempfile.TemporaryDirectory(prefix="svr_") as tmpdir:
            tmpdir = Path(tmpdir)

            src_wav = tmpdir / "src.wav"
            tgt_wav = tmpdir / "target_24000.wav"
            out_wav = tmpdir / "out.wav"
            converted_wav = tmpdir / "converted.wav"

            soundfile.write(src_wav, src, samplerate=MODEL_SR)

            # =========================
            # alpha == 0 => чистый тембр => кэшируем ref (CPU)
            # =========================
            if alpha <= eps:
                voice_key = self._voice_key_from_timbre(timbre_wave_24k, MODEL_SR, sec=10.0)

                ref_cpu = self._cache_get(voice_key)
                if ref_cpu is None:
                    # cache miss: считаем ref напрямую, без записи target wav на диск
                    ref_audio = numpy.asarray(timbre_wave_24k, numpy.float32)
                    ref_audio = ref_audio[: self.vc.DEC_COND_LEN]
                    ref_audio = pad_to_mult(ref_audio, MULT)

                    with torch.inference_mode():
                        ref_dev = self.vc.s3gen.embed_ref(ref_audio, MODEL_SR, device=self.device)

                    ref_cpu = {
                        k: (v.detach().to("cpu") if torch.is_tensor(v) else v)
                        for k, v in ref_dev.items()
                    }
                    self._cache_put(voice_key, ref_cpu)

                    # освободим временные ссылки на device-реф
                    del ref_dev
                    if str(self.device).startswith("cuda"):
                        torch.cuda.empty_cache()

                # всегда ставим ref на device (не опираемся на порядок вызовов)
                self._set_vc_ref_from_cpu(ref_cpu)

                # важно: НЕ передавать target_voice_path — иначе пересчитает ref
                wave = self.vc.generate(audio=str(src_wav), target_voice_path=None)

            # =========================
            # alpha > 0 => ref зависит от prosody => кэш НЕ используем
            # =========================
            else:
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

                tgt = numpy.asarray(mixed, numpy.float32)
                tgt = pad_to_mult(tgt, MULT)

                soundfile.write(tgt_wav, tgt, samplerate=MODEL_SR)

                # тут наоборот: передаём target_voice_path, чтобы ref считался заново
                wave = self.vc.generate(audio=str(src_wav), target_voice_path=str(tgt_wav))

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
            f"[0:a]aresample={sr},highpass=f={fc},treble=g={hi_gain_db}:f={fc}[s_hi];"
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
