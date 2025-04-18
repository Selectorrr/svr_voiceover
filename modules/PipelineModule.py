from pathlib import Path

import numpy
import soundfile
from svr_tts import SVR_TTS
from svr_tts.core import SynthesisInput
from tqdm import tqdm

from modules.AudioProcessor import AudioProcessor
from modules.CsvProcessor import CsvProcessor
from modules.SpeakerProcessor import SpeakerProcessor
from modules.TextProcessor import TextProcessor


class PipelineModule:
    def __init__(self, config):
        self.text = TextProcessor()
        self.csv = CsvProcessor(config, self.text)
        self.audio = AudioProcessor(config)
        self.speaker = SpeakerProcessor(config, self.audio)
        self.ext = config['ext']
        self.model = SVR_TTS(config['api_key'])
        self.batch_size = config['batch_size']
        pass

    def _efficient_audio_generation(self, vo_items):
        inputs = []
        for vo_item in vo_items:
            # берем наш текст и сэмплы тембра и просодии и упаковываем в задачи для синтеза
            inputs.append(SynthesisInput(text=vo_item['text'], stress=not vo_item['is_accented'],
                                         timbre_wave_24k=vo_item['style_wave_24k'],
                                         prosody_wave_24k=vo_item['raw_wave_24k']))
        # отправляем в svr tts
        waves = self.model.synthesize_batch(inputs)

        results = []
        for idx, wave in enumerate(waves):
            # не все задачи могут быть озвучены, например из за нехватки баланса или не верного текста, их пропустим
            if wave is None:
                continue
            # собираем результаты и одновременно сопоставляем с исходной громкостью и файлом
            # 22050 это фрейм рейт нейросети.
            results.append((wave, 22050, vo_items[idx]['path'], vo_items[idx]['dBFS']))
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
            dub_file = Path(str(dub_file)).with_suffix(f".{self.ext}")

            soundfile.write(dub_file, dub, sr)

    def run(self):
        # Найдем все строки что нужно озвучить с учетом разных версий файлов
        todo_records, all_records = self.csv.find_changed_text_rows_csv()

        while len(todo_records):
            todo_records = [o for o in todo_records if len(self.text.get_text(o)[0]) <= 420]
            todo_records = sorted(todo_records, key=lambda d: len(set(self.text.get_text(d)[0])), reverse=True)
            # Что-б снизить нагрузку на api сервера токенизации разобьем записи на небольшие на группы
            batches = [todo_records[i:i + self.batch_size] for i in range(0, len(todo_records), self.batch_size)]
            for records in tqdm(batches, smoothing=0, desc="Общий прогресс"):
                try:
                    # Озвучиваем группу
                    self._voice_over_record(records)
                except AssertionError as e:
                    # Неисправимая ошибка, может кончился баланс, завершаем работу
                    print(e)
                    return
            # Найдем файлы которые не удалось озвучить что-б попробовать еще раз
            todo_records, all_records = self.csv.find_changed_text_rows_csv()

        pass
