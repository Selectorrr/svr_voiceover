from functools import lru_cache
from pathlib import Path

import numpy
import soundfile


class SpeakerProcessor:
    def __init__(self, config, audio_processor):
        self.audio = audio_processor

    @lru_cache(maxsize=10)
    def _get_speaker_se_cached(self, speaker_style_cache_key, speaker_files_tuple):
        """
            Составляет сэмпл и кеширует его
        """
        speaker_files = list(speaker_files_tuple)

        style_wave_24k = self.audio.build_inference_audio(speaker_files)

        # сохраним сэмпл на диск в папку с голосами
        voice_wav = Path(f"workspace/voices/{speaker_style_cache_key}.wav")
        voice_wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(voice_wav, style_wave_24k, 24_000)

        return style_wave_24k

    def get_speaker_style(self, speaker, scope_records):
        """
            Составляет сэмпл для опеределния тембра голоса и сохраняет его на диск
        """
        speaker_style_cache_key = f"{speaker}"

        voice_pt = Path(f"workspace/voices/{speaker_style_cache_key}.wav")
        # Если сэмпл голоса уже есть то вернем его
        if voice_pt.exists():
            return soundfile.read(voice_pt)[0]

        # Для сэмпла лучше всего найти аудио файлы где спокойная речь, делаем предположение что это аудио с большим количеством букв
        speaker_files = []
        scope_records = sorted(scope_records, key=lambda x: len(x['text']), reverse=True)
        for rec in scope_records:
            speaker_files.append(rec['audio'])
        speaker_files_tuple = tuple(speaker_files)

        style_wave_24k = self._get_speaker_se_cached(speaker_style_cache_key, speaker_files_tuple)

        return style_wave_24k
