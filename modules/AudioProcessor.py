import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy
import numpy as np
import pyloudnorm
import soundfile
from filelock import FileLock
from pydub import AudioSegment
from pydub.silence import detect_silence

from modules.TextProcessor import similarity, is_tts_hallucination

OUTPUT_SR = 22_050

sound_file_formats = set(map(lambda i: f".{i}".lower(), soundfile.available_formats().keys()))

_CYR = re.compile(r"[А-Яа-яЁё]")

class AudioProcessor:
    def __init__(self, config):
        self.tone_sample_len = config['tone_sample_len']
        self.is_respect_mos = config['is_respect_mos']

    @staticmethod
    def _read_vg_in_memory(wem_file_path):
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

    @staticmethod
    def load_audio(path):
        suffix = Path(path).suffix
        if suffix in sound_file_formats:
            # Если формат файла поддерживается soundfile то читаем сразу им
            wave, sr = soundfile.read(path)
        elif suffix in {'.m4a', '.mp4', '.wma', '.aac'}:
            # Если формат поддерживается pydub то попробуем прочитать им
            wave, sr = AudioProcessor.to_ndarray(AudioSegment.from_file(path, suffix.lstrip('.'), parameters=None))
        else:
            try:
                # Что-ж видимо это экзотический формат или игровой без конвертации файл не прочитать, пробуем vgmstream
                wave, sr = AudioProcessor._read_vg_in_memory(path)
            except Exception as e:
                raise ValueError(f"Unsupported file: {e}")
        return wave, sr

    def load_audio_norm(self, path):
        wave, sr = AudioProcessor.load_audio(path)
        raw_wave, raw_sr = wave, sr
        segment = AudioProcessor.to_segment(wave, sr)
        if not AudioProcessor.is_has_sound(segment):
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
        wave, sr = self.to_ndarray(segment)
        return wave, meta, raw_wave, raw_sr

    def build_speaker_sample(self, voice_path: Path, wave_24k, mos_good=3.5, mos_bad=2):
        if self.is_respect_mos:
            mos = self.calc_mos(wave_24k, 24_000)
        else:
            mos = mos_good

        # если аудио с эффектами или сильно эмоциональное
        if mos < mos_bad:
            segment = self.to_segment(wave_24k, 24_000)
            segment = self._fix_len(segment)
            # вернем как есть без подмены на качественный
            return self.to_ndarray(segment)[0]

        lock_path = str(voice_path) + ".lock"
        lock = FileLock(lock_path, timeout=10)

        with lock:
            new_seg = self.to_segment(wave_24k, 24_000)
            if voice_path.exists():
                segment = AudioSegment.from_wav(voice_path)
                if segment.frame_rate != 24_000:
                    segment = segment.set_frame_rate(24_000)
                if len(segment) >= self.tone_sample_len:
                    return self.to_ndarray(segment)[0]
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
                tmp_path.replace(voice_path)
            return self.to_ndarray(segment)[0]

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
            segment = self.to_segment(wave, sr)
            segment = segment.set_frame_rate(16_000)
            wave, sr = self.to_ndarray(segment)
        x = wave[numpy.newaxis, :].astype(numpy.float32)
        from modules.PipelineModule import factory
        outs = factory.mos.run(None, {"input_values": x})
        mos = float(outs[0].reshape(-1).mean())
        return mos

    def validate(self, wave, sr, text_ref, iteration, threshold=0.75, hallucination_threshold=0.80):
        iteration_thr_delta = iteration * 0.05
        threshold -= iteration_thr_delta
        hallucination_threshold -= iteration_thr_delta
        if bool(re.search(r'[A-Za-z0-9]', text_ref)):
            # все равно распознать такой не сможем, редкая ситуация потому считаем валидным
            return True
        from modules.PipelineModule import factory

        segment = self.to_segment(wave, sr)
        segment = segment.set_frame_rate(16_000)
        wave_16k, _ = self.to_ndarray(segment)
        wave_16k = self.rtrim_audio(wave_16k, 16_000, top_db=40)

        asr_result = factory.asr.recognize(audio=wave_16k)
        if not bool(_CYR.search(asr_result.text)):
            return False
        if is_tts_hallucination(asr_result.text, text_ref, hallucination_threshold):
            return False
        sim = similarity(text_ref, asr_result.text)
        result = sim >= threshold
        # if not result:
        #     print(f"invalid sim: asr {asr_result.text} text: {text_ref}")
        # else:
            # print(f"valid sim")
        return result

    @staticmethod
    def to_segment(y, sr=24000):
        memory_buffer = io.BytesIO()
        soundfile.write(memory_buffer, y, samplerate=sr, format='WAV')
        memory_buffer.seek(0)
        return AudioSegment(memory_buffer)

    @staticmethod
    def to_ndarray(segment):
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

    def pad_with_silent(self, raw_wave, raw_sr):
        target_length = self.audio_len(raw_wave, raw_sr)
        silent_len = max(1 - target_length, 0.1) * 1000
        raw_segment = self.to_segment(raw_wave, raw_sr) + AudioSegment.silent(duration=silent_len, frame_rate=raw_sr)

        raw_wave, raw_sr = self.to_ndarray(raw_segment)
        return raw_wave

    def restore_meta(self, wave, sr, meta):
        segment = self.to_segment(wave, sr)
        segment = segment.set_frame_rate(meta['frame_rate'])
        segment = segment.set_sample_width(meta['sample_width'])
        segment = segment.apply_gain(meta['dBFS'] - segment.dBFS)
        segment = segment.set_channels(meta['channels'])
        wave, sr = self.to_ndarray(segment)
        return wave, sr

    def prepare_prosody(self, wave, sr, top_db=40):
        # обрезаем тишину справа
        wave = self.rtrim_audio(wave, sr, top_db=top_db)
        # органичиваем максимальный размер
        # wave = wave[:MAX_PROSODY_LEN * sr]
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

    @staticmethod
    def _dur_sec(wave: np.ndarray, sr: int) -> float:
        if wave is None or sr <= 0:
            return 0.0
        n = int(wave.shape[0])
        return n / float(sr) if n > 0 else 0.0

    @staticmethod
    def speedup_if_need(dub, sr, i_raw_wave, i_raw_sr, max_extra_speed=0.40):
        """
        dub ускоряем так, чтобы уложиться в длительность i_raw_wave.
        Ускорение за один раз ограничиваем max_extra_speed.
        """
        if dub is None or i_raw_wave is None:
            return dub

        dub_len = AudioProcessor._dur_sec(dub, sr)
        raw_len = AudioProcessor._dur_sec(i_raw_wave, i_raw_sr)

        if raw_len <= 0 or dub_len <= 0:
            return dub

        ratio = dub_len / raw_len

        if ratio <= 1.0:
            return dub  # и так не длиннее

        ratio = min(ratio, 1.0 + max_extra_speed)

        segment = AudioProcessor.to_segment(dub, sr)
        memory_buffer = io.BytesIO()
        segment.export(memory_buffer, format='wav', parameters=["-af", f"atempo={ratio}"])
        memory_buffer.seek(0)
        wave, sr = soundfile.read(memory_buffer)
        return wave