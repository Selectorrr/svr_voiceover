import re

import numpy as np
import onnx_asr
from onnx_asr.asr import TimestampedResult

from modules.TextProcessor import similarity


class AsrModel:
    def __init__(self, providers, provider_options):
        self.session = onnx_asr.load_model("alphacep/vosk-model-ru", providers=providers, provider_options=provider_options).with_timestamps()

    def recognize(
            self,
            audio: np.ndarray,

    ):
        result = self.session.recognize(audio, language="ru")
        return result


    # получаем список (word, end_time_sec) из ASR
    def _words_with_end_times(self, asr: TimestampedResult):
        # восстановим полный текст (обычно = asr.text, но берём из tokens для надёжной длины)
        full = ''.join(asr.tokens) if asr.tokens else asr.text

        # соответствие позиции символа -> индекса токена
        cum = []
        total = 0
        for t in asr.tokens:
            total += len(t)
            cum.append(total)  # длина текста после этого токена

        # находим слова и их end_time: конец слова попадает в некий token_index
        words = []
        for m in re.finditer(r'[A-Za-zА-Яа-яЁё0-9]+', full):
            end_char = m.end()  # позиция конца слова в символах
            # ищем первый cum >= end_char
            lo, hi = 0, len(cum) - 1
            token_idx = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                if cum[mid] >= end_char:
                    token_idx = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            end_time = asr.timestamps[token_idx] if asr.timestamps else 0.0
            words.append((m.group(0), end_time))
        return words

    def end_time(self, asr: TimestampedResult, ref_text: str, threshold: float = 0.75) -> int | None:
        # берём ПОСЛЕДНЕЕ слово из эталонного текста
        ref_words = re.findall(r'[A-Za-zА-Яа-яЁё0-9]+', ref_text)
        if not ref_words:
            return None

        ref_last = ref_words[-1]

        words = self._words_with_end_times(asr)

        if not len(words):
            return None
        w, t = words[-1]
        if len(words) < 2:
            return int(t * 1000)
        prev_w, prev_t = words[-2]

        if not similarity(ref_last, w) >= threshold and similarity(ref_last, prev_w) >= threshold:
            #поймали галюцинацию распознанную как слово
            return int(prev_t * 1000)
        else:
            return int(t * 1000)