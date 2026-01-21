import hashlib
import logging
import tempfile
import warnings

logging.basicConfig(level=logging.INFO)
# torchaudio: ругается на backend=...
warnings.filterwarnings(
    "ignore",
    message=r".*'backend' parameter is not used by TorchCodec AudioDecoder.*",
    category=UserWarning,
)

# cosyvoice: deprecated autocast
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)
from pathlib import Path

import numpy
import numpy as np
import soundfile
import torch

from vendors.CosyVoice.cosyvoice.cli.cosyvoice import AutoModel

MODEL_SR = 24_000
_FADE_SAMPLES = int(24000 * 0.1)  # 1200


class VcCV3:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.vc = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

        self.cache_dir = Path("workspace/voices/timbre_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, full_wave, timbre_wave_24k_orig, prosody_wave_24k) -> numpy.ndarray:
        with tempfile.TemporaryDirectory(prefix="svr_") as tmpdir:
            tmpdir = Path(tmpdir)
            prompt_wav = tmpdir / "prompt.wav"
            timbre_wav = tmpdir / "timbre.wav"
            source_wav = tmpdir / "source.wav"
            soundfile.write(source_wav, full_wave, samplerate=MODEL_SR)
            soundfile.write(timbre_wav, timbre_wave_24k_orig, samplerate=MODEL_SR)
            soundfile.write(prompt_wav, prosody_wave_24k, samplerate=MODEL_SR)
            result = next(self.inference_vc(source_wav, prompt_wav, timbre_wav, stream=False, speed=1.0))
            result = result['tts_speech'].cpu().squeeze().numpy()
            result = self._fade_in_out(result)
            return result

    def frontend_vc(self, source_speech_16k, prompt_wav, timbre_wav):
        prompt_speech_token, prompt_speech_token_len = self.vc.frontend._extract_speech_token(prompt_wav)
        prompt_speech_feat, prompt_speech_feat_len = self.vc.frontend._extract_speech_feat(prompt_wav)
        embedding = self._build_speaker_emb(timbre_wav)
        source_speech_token, source_speech_token_len = self.vc.frontend._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token,
                       'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input

    def inference_vc(self, source_wav, prompt_wav, timbre_wav, stream=False, speed=1.0):
        model_input = self.frontend_vc(source_wav, prompt_wav, timbre_wav)
        for model_output in self.vc.model.tts(**model_input, stream=stream, speed=speed):
            yield model_output

    def _load_spk_embedding(self, key: str) -> torch.Tensor | None:
        p = self.cache_dir / f"{key}.pt"
        if not p.exists():
            return None
        try:
            t = torch.load(p, map_location="cpu")
            if torch.is_tensor(t):
                return t
        except Exception:
            pass
        return None

    def _save_speaker_emb(self, key: str, spk_embedding: torch.Tensor):
        p = self.cache_dir / f"{key}.pt"
        tmp = p.with_suffix(".pt.tmp")
        torch.save(spk_embedding, tmp)
        tmp.replace(p)

    def _build_speaker_emb(self, prompt_timbre_path):
        prompt_timbre, sr = soundfile.read(prompt_timbre_path)
        key = self._voice_key_from_timbre(prompt_timbre, sr, sec=10.0)
        spk_embedding = self._load_spk_embedding(key)
        if spk_embedding is None:
            spk_embedding = self.vc.frontend._extract_spk_embedding(prompt_timbre_path)
            self._save_speaker_emb(key, spk_embedding)
        return spk_embedding

    @staticmethod
    def _voice_key_from_timbre(timbre_wave_24k: numpy.ndarray, sr: int, sec: float = 10.0) -> str:
        n = int(sr * sec)
        x = numpy.asarray(timbre_wave_24k[:n], dtype=numpy.float32)
        h = hashlib.blake2b(digest_size=16)
        h.update(str(sr).encode("utf-8"))
        h.update(x.tobytes())
        return h.hexdigest()

    @staticmethod
    def _fade_in_out(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).copy()
        n = _FADE_SAMPLES
        if x.size < 2 * n:
            n = x.size // 2
        if n <= 0:
            return x

        fade_in = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, n, endpoint=False, dtype=np.float32)

        x[:n] *= fade_in
        x[-n:] *= fade_out
        return x
