import difflib
import io
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy
import numpy as np
import pyloudnorm
import soundfile
from filelock import FileLock, Timeout
from pydub import AudioSegment
from pydub.silence import detect_silence

MAX_PROSODY_LEN = 20


class AudioProcessor:
    def __init__(self, config):
        self.tone_sample_len = config['tone_sample_len']
        self.sound_file_formats = set(map(lambda i: f".{i}".lower(), soundfile.available_formats().keys()))
        self.is_respect_mos = config['is_respect_mos']
        self.is_use_voice_len = config['is_use_voice_len']

        pass

    def _read_vg_in_memory(self, wem_file_path):
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav = temp_file.name

        try:
            # Конвертируем WEM в WAV через vgmstream
            process = subprocess.run(
                ["vgmstream-cli", "-i", wem_file_path, "-o", temp_wav],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                raise RuntimeError(f"Ошибка конвертации: {process.stderr.decode()}")

            # Загружаем WAV в память
            audio, sr = soundfile.read(temp_wav)
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        return audio, sr

    def _normalize_audio_lufs(self, audio_segment, target_lufs=-16.0):
        """Нормализует аудио до заданного уровня громкости (LUFS)."""
        # Получаем сэмплы и преобразуем в numpy-массив
        samples = numpy.array(audio_segment.get_array_of_samples())

        sample_rate = audio_segment.frame_rate
        channels = audio_segment.channels
        sample_width = audio_segment.sample_width  # В байтах

        # Преобразуем сэмплы в диапазон от -1.0 до 1.0
        max_value = float(int((2 ** (8 * sample_width)) / 2))
        samples = samples / max_value

        # Если аудио стерео, преобразуем сэмплы в 2D-массив
        if channels > 1:
            samples = samples.reshape((-1, channels))

        # Создаём объект Meter для измерения громкости
        meter = pyloudnorm.Meter(sample_rate, block_size=0.2)
        loudness = meter.integrated_loudness(samples.astype(numpy.float32))

        # Вычисляем необходимое изменение громкости
        loudness_difference = target_lufs - loudness

        # Применяем изменение громкости
        normalized_audio = audio_segment.apply_gain(loudness_difference)
        return normalized_audio

    def load_audio(self, path):
        suffix = Path(path).suffix
        if suffix in self.sound_file_formats:
            # Если формат файла поддерживается soundfile то читаем сразу им
            wave, sr = soundfile.read(path)
        elif suffix in {'.m4a', '.mp4', '.wma', '.aac'}:
            # Если формат поддерживается pydub то попробуем прочитать им
            wave, sr = self._to_ndarray(AudioSegment.from_file(path, suffix.lstrip('.'), parameters=None))
        else:
            try:
                # Что-ж видимо это экзотический формат или игровой без конвертации файл не прочитать, пробуем vgmstream
                wave, sr = self._read_vg_in_memory(path)
            except Exception as e:
                raise ValueError(f"Unsupported file: {e}")
        raw_wave, raw_sr = wave, sr
        segment = self._to_segment(wave, sr)
        if not self.is_has_sound(segment):
            raise ValueError(f"No sound: {path}")

        channels = segment.channels
        if channels > 1:
            # делаем моно потому что нейросеть работает с моно
            segment = segment.set_channels(1)
        # запомним оригинальные характеристики звука
        meta = {
            'frame_rate': segment.frame_rate,
            'sample_width': segment.sample_width,
            'dBFS': segment.dBFS,
            'channels': channels
        }
        # и приведем к нормальной потому что нейросеть обучалась на нормализованной громкости
        segment = self._normalize_audio_lufs(segment)
        if segment.frame_rate != 24_000:
            segment = segment.set_frame_rate(24_000)
        wave, sr = self._to_ndarray(segment)
        return wave, meta, raw_wave, raw_sr

    def build_speaker_sample(self, voice_path: Path, wave_24k, mos_good=3.5, mos_bad=2):
        if self.is_respect_mos:
            mos = self.calc_mos(wave_24k, 24_000)
        else:
            mos = mos_good

        # если аудио с эффектами или сильно эмоциональное
        if mos < mos_bad:
            segment = self._to_segment(wave_24k, 24_000)
            segment = self._fix_len(segment)
            # вернем как есть без подмены на качественный
            return self._to_ndarray(segment)[0]

        lock_path = str(voice_path) + ".lock"

        if os.path.exists(lock_path):
            age = time.time() - os.path.getmtime(lock_path)
            if age > 10:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

        lock = FileLock(lock_path, timeout=10)
        with lock:
            new_seg = self._to_segment(wave_24k, 24_000)
            if voice_path.exists():
                segment = AudioSegment.from_wav(voice_path)
                if len(segment) >= self.tone_sample_len:
                    return self._to_ndarray(segment)[0]
                if mos >= mos_good:
                    end_slice = segment[-len(new_seg):] if len(segment) >= len(new_seg) else AudioSegment.empty()
                    if not self._is_similar(end_slice, new_seg):
                        segment += new_seg
            else:
                segment = new_seg

            segment = self._fix_len(segment)

            if mos >= mos_good:
                voice_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = voice_path.with_suffix(".wav.tmp")
                segment.export(tmp_path, format="wav")
                tmp_path.replace(voice_path)  # atomic rename
            return self._to_ndarray(segment)[0]

    def _fix_len(self, segment):
        while len(segment) < 1000:
            segment += segment
        segment = segment[:self.tone_sample_len]
        return segment

    def _is_similar(self, seg1: AudioSegment, seg2: AudioSegment, thresh=0.9) -> bool:
        a = numpy.array(seg1.get_array_of_samples(), dtype=float)
        b = numpy.array(seg2.get_array_of_samples(), dtype=float)
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        sim = numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b) + 1e-8)
        return sim >= thresh

    def calc_mos(self, wave, sr):
        if sr != 16_000:
            segment = self._to_segment(wave, sr)
            segment = segment.set_frame_rate(16_000)
            wave, sr = self._to_ndarray(segment)
        x = wave[numpy.newaxis, :].astype(numpy.float32)
        from modules.PipelineModule import factory
        outs = factory.mos.run(None, {"input_values": x})
        mos = float(outs[0].reshape(-1).mean())
        return mos

    def validate(self, wave, sr, text_ref, threshold=0.75):
        if bool(re.search(r'[A-Za-z0-9]', text_ref)):
            # все равно распознать такой не сможем, редкая ситуация потому считаем валидным
            return True
        from modules.PipelineModule import factory
        if sr != 16_000:
            segment = self._to_segment(wave, sr)
            segment = segment.set_frame_rate(16_000)
            wave, sr = self._to_ndarray(segment)
        outs = factory.asr(audio=wave)
        similarity = self._similarity(text_ref, outs)
        return similarity >= threshold

    def _normalize(self, text):
        text = text.lower().replace('ё', 'е').replace('й', 'и')
        # нижний регистр, убрать пунктуацию, пробелы
        text = re.sub(r'[^\w]', '', text)
        # удалить повторяющиеся подряд символы (например: "оооо" → "о")
        text = re.sub(r'(.)\1+', r'\1', text)
        return text

    def _similarity(self, a, b):
        return difflib.SequenceMatcher(None, self._normalize(a), self._normalize(b)).ratio()

    @staticmethod
    def _to_segment(y, sr=24000):
        memory_buffer = io.BytesIO()
        soundfile.write(memory_buffer, y, samplerate=sr, format='WAV')
        memory_buffer.seek(0)
        return AudioSegment(memory_buffer)

    @staticmethod
    def _to_ndarray(segment):
        memory_buffer = io.BytesIO()
        segment.export(memory_buffer, format='wav')
        memory_buffer.seek(0)
        data, sr = soundfile.read(memory_buffer)
        return data, sr

    @staticmethod
    def audio_len(wav, sr):
        audio_length = len(wav) / sr
        return audio_length

    def rtrim_audio(self, y, sr, lower_bound=1, top_db=40):
        orig_len = self.audio_len(y, sr)
        if not lower_bound or orig_len <= lower_bound:
            return y
        _, index = librosa.effects.trim(y, top_db=top_db)
        y = y[:index[1]]
        return y

    def voice_len(self, y, sr, lower_bound=1):
        y = self.rtrim_audio(y, sr, lower_bound)
        r_len = self.audio_len(y, sr)
        return r_len

    def pad_with_silent(self, raw_wave, raw_sr):
        target_length = self.audio_len(raw_wave, raw_sr)
        silent_len = max(1 - target_length, 0.1) * 1000
        raw_segment = self._to_segment(raw_wave, raw_sr) + AudioSegment.silent(duration=silent_len, frame_rate=raw_sr)

        raw_wave, raw_sr = self._to_ndarray(raw_segment)
        return raw_wave

    def get_sound_center(self, signal, top_db=40):
        if signal.ndim > 1:
            mono = numpy.mean(signal, axis=tuple(range(1, signal.ndim)))
        else:
            mono = signal
        _, idx = librosa.effects.trim(mono, top_db=top_db)
        start, end = idx
        return start

    def align_by_samples(self, wave, wave_sr, raw_wave, raw_sr, top_db=40):
        if wave_sr != raw_sr:
            raw_wave, raw_sr = self._to_ndarray(self._to_segment(raw_wave, raw_sr).set_frame_rate(wave_sr))

        orig_len = raw_wave.shape[0]
        if self.is_use_voice_len:
            target_len = int(self.voice_len(raw_wave, raw_sr) * raw_sr)
        else:
            target_len = orig_len

        start_wave = self.get_sound_center(wave, top_db)
        start_raw = self.get_sound_center(raw_wave, top_db)

        desired_shift = start_raw - start_wave

        max_left_shift = start_wave
        max_right_shift = target_len - (wave.shape[0] - start_wave)

        # Ограничиваем shift, чтобы полезный сигнал полностью влез
        safe_shift = int(numpy.clip(desired_shift, -max_left_shift, max_right_shift))

        # Применяем сдвиг
        if safe_shift > 0:
            pad = numpy.zeros((safe_shift,) + wave.shape[1:], dtype=wave.dtype)
            wave = numpy.concatenate([pad, wave], axis=0)
        elif safe_shift < 0:
            wave = wave[-safe_shift:]

        # Дополняем справа до нужной длины
        if self.is_use_voice_len:
            wave = self.pad_or_trim_to_len(target_len, wave)
        # и в любом случае возвращаем длину orig_len
        wave = self.pad_or_trim_to_len(orig_len, wave)
        return wave

    def pad_or_trim_to_len(self, target_len, wave):
        if wave.shape[0] < target_len:
            pad = numpy.zeros((target_len - wave.shape[0],) + wave.shape[1:], dtype=wave.dtype)
            wave = numpy.concatenate([wave, pad], axis=0)
        else:
            wave = wave[:target_len]
        return wave

    def restore_meta(self, wave, sr, meta):
        segment = self._to_segment(wave, sr)
        segment = segment.set_frame_rate(meta['frame_rate'])
        segment = segment.set_sample_width(meta['sample_width'])
        segment = segment.apply_gain(meta['dBFS'] - segment.dBFS)
        segment = segment.set_channels(meta['channels'])
        wave, sr = self._to_ndarray(segment)
        return wave, sr

    def speedup(self, wave, sr, ratio, max_len, increment=0.05):
        segment = self._to_segment(wave, sr)
        ratio = min(2 - increment, ratio)
        ratio += increment
        while len(segment) > max_len * 1000:
            memory_buffer = io.BytesIO()
            # эксперементы показали что atempo дает наименьшие артефакты при ускорении
            segment.export(memory_buffer, format='wav', parameters=["-af", f"atempo={ratio}"])
            memory_buffer.seek(0)
            wave, sr = soundfile.read(memory_buffer)
            segment = self._to_segment(wave, sr)
            ratio = 1 + increment

        wave, sr = self._to_ndarray(segment)
        return wave

    def mixing(self, dub_wave, dub_sr, raw_wave, raw_sr):
        dub_segment = self._to_segment(dub_wave, dub_sr)
        resource_segment = self._to_segment(raw_wave, raw_sr)
        raw_audio = resource_segment - 7
        if raw_audio.frame_rate != dub_segment.frame_rate:
            raw_audio = raw_audio.set_frame_rate(dub_segment.frame_rate)

        # Выравнивание длительности аудио
        max_duration = max(len(raw_audio), len(dub_segment))
        raw_audio = raw_audio + AudioSegment.silent(duration=max_duration - len(raw_audio))
        dub_segment = dub_segment + AudioSegment.silent(duration=max_duration - len(dub_segment))

        # Смешивание
        mixed_audio = raw_audio.overlay(dub_segment)
        return self._to_ndarray(mixed_audio)

    @staticmethod
    def trimmed_audio_len(wave, sr, top_db=40):
        raw_wave, _ = librosa.effects.trim(wave, top_db=top_db)
        return len(raw_wave) / sr

    def prepare_prosody(self, wave, sr, top_db=40):
        # обрезаем тишину справа
        wave = self.rtrim_audio(wave, sr, top_db=top_db)
        # органичиваем максимальный размер
        wave = wave[:MAX_PROSODY_LEN * sr]
        keep = int(0.1 * sr)  # 100 мс
        # убираем последние 100 мс тк добавим ниже тишины
        wave = wave[:-keep]
        # фейдим новые последние 100 мс что-б не было щелчка
        start_fade = max(len(wave) - keep, 0)
        ramp = np.linspace(1.0, 0.0, keep, endpoint=True)
        wave[start_fade:] *= ramp
        # дополняем до 1 сек или 100 мс тишины
        wave = self.pad_with_silent(wave, sr)
        return wave

    @staticmethod
    def is_has_sound(seg, silence_thresh=-40, min_silence_len=100):
        silent_ranges = detect_silence(
            seg,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        total_silence = sum(end - start for start, end in silent_ranges)
        return total_silence < len(seg)
