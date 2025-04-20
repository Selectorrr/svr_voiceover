from multiprocessing import get_context
from pathlib import Path

import librosa
import numpy
import onnxruntime
import soundfile
from pqdm.processes import pqdm
from svr_tts.core import SynthesisInput

from modules.AudioProcessor import AudioProcessor
from modules.CsvProcessor import CsvProcessor
from modules.ModelFactory import ModelFactory
from modules.SpeakerProcessor import SpeakerProcessor
from modules.TextProcessor import TextProcessor

onnxruntime.set_default_logger_severity(3)
factory = None


def _worker_init(config):
    global factory
    factory = ModelFactory(config)


class PipelineModule:
    def __init__(self, config):
        self.text = TextProcessor()
        self.csv = CsvProcessor(config, self.text)
        self.audio = AudioProcessor(config)
        self.speaker = SpeakerProcessor(self.audio)
        self.config = config

    def _efficient_audio_generation(self, vo_items):
        inputs = []
        for vo_item in vo_items:
            # берем наш текст и сэмплы тембра и просодии и упаковываем в задачи для синтеза
            inputs.append(SynthesisInput(text=vo_item['text'], stress=not vo_item['is_accented'],
                                         timbre_wave_24k=vo_item['style_wave_24k'],
                                         prosody_wave_24k=vo_item['raw_wave_24k']))
        # отправляем в svr tts
        job_n = ModelFactory.get_job_n()
        # noinspection PyUnresolvedReferences
        waves = factory.svr_tts.synthesize_batch(inputs, tqdm_kwargs={'leave': False, 'smoothing': 0,
                                                                      'position': job_n + 1,
                                                                      'postfix': f"job_n {job_n}"})

        results = []
        for idx, wave in enumerate(waves):
            # не все задачи могут быть озвучены, например из за нехватки баланса или не верного текста, их пропустим
            if wave is None:
                continue
            # собираем результаты и одновременно сопоставляем с исходной громкостью и файлом
            # 22050 это фрейм рейт нейросети.
            vo_item = vo_items[idx]

            # если длина аудио строго должна совпадать с оригиналом
            if self.config['is_strict_len']:
                raw_len = len(vo_item['raw_wave_24k']) / 24_000
                # уберем тишину тк это безболезненно
                wave, _ = librosa.effects.trim(wave, top_db=40)
                wave_len = len(wave) / 22050
                rate = wave_len / raw_len
                if rate > 1:
                    # ускорим, но не сильнее чем на 50 процентов
                    wave = self.audio.speedup(wave, 22050, min(1.5, rate), raw_len)
            results.append((wave, 22050, vo_item['path'], vo_item['dBFS']))
        return results

    def _voice_over_items(self, vo_items):
        for vo_item in vo_items:
            # сэмпл просодии не может быть короче 1й секунды, дополняем тишиной если надо
            vo_item['raw_wave_24k'] = self.audio.pad_with_silent(vo_item['raw_wave_24k'], 24_000)
            # нейросеть для чисел с плавающей точкой использует 32 бита, потому приводим аудио в этот формат
            vo_item['style_wave_24k'] = vo_item['style_wave_24k'].astype(numpy.float32)
            vo_item['raw_wave_24k'] = vo_item['raw_wave_24k'].astype(numpy.float32)

        results = self._efficient_audio_generation(vo_items)
        return results

    def _voice_over_record(self, records):
        """
            Озвучивает группу записей
        """
        vo_items = []
        for record in records:
            path = Path(f"workspace/resources/{record['audio']}")
            # Прочитаем оригинальный аудио файл и его громкость для определения просодии
            raw_wave_24k, dBFS = self.audio.load_audio(str(path))
            # Определяем текст для озвучки
            text, is_accented = self.text.get_text(record)

            # Возьмем все строки которые произносит этот персонаж для того что-б определить его тембр
            # Подготовим сэмпл тембра голоса
            style_wave_24k = self.speaker.get_speaker_style(record['speaker'], raw_wave_24k)
            # соберем все у кучку
            vo_item = {
                'text': text,
                'is_accented': is_accented,
                'raw_wave_24k': raw_wave_24k,
                'dBFS': dBFS,
                'style_wave_24k': style_wave_24k,
                'path': record['audio']
            }
            vo_items.append(vo_item)

        # Озвучиваем батч
        results = self._voice_over_items(vo_items)

        for dub, sr, i_path, i_dBFS in results:
            # Восстановим громкость оригинального аудио
            dub = self.audio.restore_loudness(dub, sr, i_dBFS)
            # Сохраняем итоговый файл
            dub_file = Path(f"workspace/dub/{i_path}")
            dub_file.parent.mkdir(parents=True, exist_ok=True)
            dub_file = Path(str(dub_file)).with_suffix(f".{self.config['ext']}")

            soundfile.write(dub_file, dub, sr)

    def run(self):
        # Найдем все строки что нужно озвучить с учетом разных версий файлов
        todo_records, all_records = self.csv.find_changed_text_rows_csv()

        todo_records = [o for o in todo_records if len(self.text.get_text(o)[0]) <= 420]
        todo_records = sorted(todo_records, key=lambda d: len(set(self.text.get_text(d)[0])), reverse=True)
        # Что-б снизить нагрузку на api сервера токенизации разобьем записи на небольшие на группы
        batches = [todo_records[i:i + self.config['batch_size']] for i in
                   range(0, len(todo_records), self.config['batch_size'])]

        mp_context = get_context('spawn')

        # если однопроцессный режим то надо вручную инициализировать модельки
        if self.config['n_jobs'] == 1:
            _worker_init(self.config)

        try:
            pqdm(batches, self._voice_over_record,
                 n_jobs=self.config['n_jobs'],
                 mp_context=mp_context,
                 smoothing=0,
                 desc='Общий прогресс',
                 initializer=_worker_init,
                 initargs=(self.config,),
                 position=0,
                 exception_behaviour='immediate'
                 )
        except AssertionError as e:
            # Неисправимая ошибка, может кончился баланс, завершаем работу
            print(e)
