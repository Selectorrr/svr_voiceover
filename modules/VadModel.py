import warnings
from typing import Callable

import numpy as np
import onnxruntime


class VadModel:
    def __init__(self, providers, provider_options):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession('models/silero_vad.onnx', providers=providers,
                                                    sess_options=opts, provider_options=provider_options)
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x: np.ndarray, sr: int):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = None
        self._last_sr = 0
        self._last_batch_size = 0

    def run(self, x: np.ndarray, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size or self._last_sr != sr or self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if self._context is None:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)

        if sr in [8000, 16000]:
            ort_inputs = {
                'input': x.astype(np.float32),
                'state': self._state.astype(np.float32),
                'sr': np.array(sr, dtype=np.int64)
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out

    def __call__(
            self,
            audio: np.ndarray,
            threshold: float = 0.5,
            sampling_rate: int = 16000,
            min_speech_duration_ms: int = 250,
            max_speech_duration_s: float = float('inf'),
            min_silence_duration_ms: int = 100,
            speech_pad_ms: int = 30,
            return_seconds: bool = False,
            time_resolution: int = 1,
            progress_tracking_callback: Callable[[float], None] = None,
            neg_threshold: float = None,
    ):
        # проверяем все размеры, убираем лишние оси
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.array(audio, dtype=float)
            except:
                raise TypeError("Audio cannot be cast to numpy array. Cast it manually")
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            raise ValueError("Audio must be 1-D")

        # понижаем сэмплинг, если нужно
        if sampling_rate > 16000 and sampling_rate % 16000 == 0:
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
            warnings.warn('Sampling rate is a multiple of 16000, casting to 16000!')
        else:
            step = 1

        if sampling_rate not in [8000, 16000]:
            raise ValueError("Supported sample rates: 8000 or 16000 (or multiple of 16000)")

        window_size = 512 if sampling_rate == 16000 else 256
        self.reset_states()
        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        max_speech_samples = sampling_rate * max_speech_duration_s - window_size - 2 * speech_pad_samples
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        min_silence_at_max = sampling_rate * 98 / 1000

        L = len(audio)
        speech_probs = []
        # вычисляем вероятности речи
        for start in range(0, L, window_size):
            chunk = audio[start:start + window_size]
            if len(chunk) < window_size:
                pad_len = window_size - len(chunk)
                chunk = np.pad(chunk, (0, pad_len), mode='constant', constant_values=0)
            prob = float(self.run(chunk[np.newaxis, :], sampling_rate)[0])
            speech_probs.append(prob)

            # прогресс
            if progress_tracking_callback:
                progress = min(start + window_size, L)
                progress_tracking_callback(progress / L * 100)

        # сегментация точно как в оригинале
        neg_threshold = neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
        triggered = False
        speeches = []
        current = {}
        temp_end = prev_end = next_start = 0

        for i, p in enumerate(speech_probs):
            pos = i * window_size
            if p >= threshold and not triggered:
                triggered = True
                current['start'] = pos
            elif triggered and (p < neg_threshold):
                if temp_end == 0:
                    temp_end = pos
                if pos - temp_end > min_silence_at_max:
                    prev_end = temp_end
                if pos - temp_end >= min_silence_samples:
                    current['end'] = temp_end
                    if current['end'] - current['start'] > min_speech_samples:
                        speeches.append(current)
                    current = {}
                    triggered = False
                    temp_end = prev_end = next_start = 0

            # обрезаем слишком длинные
            if triggered and pos - current['start'] > max_speech_samples:
                end = prev_end or pos
                current['end'] = end
                speeches.append(current)
                current = {}
                triggered = False
                temp_end = prev_end = next_start = 0

        # последний сегмент
        if triggered and L - current.get('start', 0) > min_speech_samples:
            current['end'] = L
            speeches.append(current)

        # паддинг и коррекция границ
        for idx, seg in enumerate(speeches):
            if idx == 0:
                seg['start'] = max(0, seg['start'] - speech_pad_samples)
            end = seg['end']
            if idx < len(speeches) - 1:
                nxt = speeches[idx + 1]['start']
                silence = nxt - end
                if silence < 2 * speech_pad_samples:
                    seg['end'] += silence // 2
                    speeches[idx + 1]['start'] = max(0, nxt - silence // 2)
                else:
                    seg['end'] = min(L, end + speech_pad_samples)
                    speeches[idx + 1]['start'] = max(0, nxt - speech_pad_samples)
            else:
                seg['end'] = min(L, end + speech_pad_samples)

        # в секундах или обратно в оригинальную частоту
        if return_seconds:
            dur = sampling_rate
            for seg in speeches:
                seg['start'] = round(seg['start'] / dur, time_resolution)
                seg['end'] = round(seg['end'] / dur, time_resolution)
        elif step > 1:
            for seg in speeches:
                seg['start'] *= step
                seg['end'] *= step

        return speeches
