import io
import os
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy
import pyloudnorm
import soundfile
from filelock import FileLock, Timeout
from pydub import AudioSegment


class AudioProcessor:
    def __init__(self, config):
        self.tone_sample_len = config['tone_sample_len']
        self.max_speed_ratio = config['max_speed_ratio']
        self.sound_file_formats = set(map(lambda i: f".{i}".lower(), soundfile.available_formats().keys()))

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

        segment = self._to_segment(wave, sr)

        channels = segment.channels
        if channels > 1:
            # делаем моно потому что нейросеть работает с моно
            segment = segment.set_channels(1)
        # запомним оригинальную громкость
        dBFS = segment.dBFS
        # и приведем к нормальной потому что нейросеть обучалась на нормализованной громкости
        segment = self._normalize_audio_lufs(segment)
        if segment.frame_rate != 24_000:
            segment = segment.set_frame_rate(24_000)
        wave, sr = self._to_ndarray(segment)
        return wave, dBFS

    def build_speaker_sample(self, voice_path: Path, wave_24k, mos_good=3.5, mos_bad=2):
        mos = self.calc_mos(wave_24k, 24_000)

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
        try:
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
        except Timeout:
            raise RuntimeError(f"Не удалось взять блокировку {lock_path} за 5 сек")
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)

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
    def _audio_len(wav, sr):
        audio_length = len(wav) / sr
        return audio_length

    def pad_with_silent(self, raw_wave, raw_sr):
        target_length = self._audio_len(raw_wave, raw_sr)
        silent_len = max(1 - target_length, 0.1) * 1000
        raw_segment = self._to_segment(raw_wave, raw_sr) + AudioSegment.silent(duration=silent_len, frame_rate=raw_sr)

        raw_wave, raw_sr = self._to_ndarray(raw_segment)
        return raw_wave

    def prepare_prosody_wave(self, raw_wave, raw_sr):
        segment_orig = self._to_segment(raw_wave, raw_sr)
        segment = segment_orig
        # нужна одна секунда звука минимум
        while int(len(librosa.effects.trim(self._to_ndarray(segment)[0], top_db=40)[0]) / raw_sr * 1000) < 1000:
            segment += segment_orig
        if raw_sr != 24000:
            segment = segment.set_frame_rate(24_000)
        raw_wave, raw_sr = self._to_ndarray(segment)
        raw_wave, _ = librosa.effects.trim(raw_wave, top_db=40)
        # 100ms тишины в конце
        raw_wave = numpy.pad(raw_wave, (0, int(0.1 * raw_sr)), 'constant')
        return raw_wave.astype(numpy.float32)

    def restore_loudness(self, wave, sr, dBFS):
        segment = self._to_segment(wave, sr)
        segment = segment.apply_gain(dBFS - segment.dBFS)
        wave, sr = self._to_ndarray(segment)
        return wave

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
