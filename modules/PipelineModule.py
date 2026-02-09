import csv
import traceback
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy
import onnxruntime
from pqdm.processes import pqdm
from svr_tts.core import SynthesisInput
from tqdm.auto import tqdm as _tqdm

from modules.AudioProcessor import AudioProcessor
from modules.CsvProcessor import CsvProcessor
from modules.ModelFactory import ModelFactory
from modules.SpeakerProcessor import SpeakerProcessor
from modules.TextProcessor import TextProcessor

INPUT_SR = 24_000
OUTPUT_SR = 22_050

FADE_LEN = int(0.1 * OUTPUT_SR)

onnxruntime.set_default_logger_severity(3)
factory = None
_progress_counter = None


def _worker_init(config, progress_counter=None):
    """Initializer for pqdm workers."""
    global factory, _progress_counter
    factory = ModelFactory(config)
    _progress_counter = progress_counter


class _CharSyncTqdm(_tqdm):
    """tqdm that syncs its current position from a shared counter."""

    def __init__(self, *args, progress_counter=None, **kwargs):
        self._c = progress_counter
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        if self._c is None:
            return super().update(n)
        self.n = self._c.value
        self.refresh()
        return True


class PipelineModule:
    def __init__(self, config):
        self.text = TextProcessor()
        self.csv = CsvProcessor(config, self.text)
        self.audio = AudioProcessor(config)
        self.speaker = SpeakerProcessor(self.audio)
        self.config = config
        self.iteration = 0

    def synthesize_batch_with_split(self, inputs: list[SynthesisInput]):
        job_n = ModelFactory.get_job_n()

        # noinspection PyUnresolvedReferences
        try:
            all_waves = factory.svr_tts.synthesize(
                inputs,
                tqdm_kwargs={
                    'leave': False,
                    'smoothing': 0,
                    'position': job_n + 1,
                    'postfix': f"job_n {job_n}",
                    'dynamic_ncols': True
                },
                rtrim_top_db=15,
                stress_exclusions=self.csv.load_stress_exclusions()
            )
        except Exception:
            traceback.print_exc()
            all_waves = [None] * len(inputs)

        result = [None] * len(inputs)

        for i, full_wave in enumerate(all_waves):
            if full_wave is None:
                continue

            is_valid = None
            try:
                try:
                    is_valid = self.audio.validate(
                        full_wave, factory.svr_tts.OUTPUT_SR, inputs[i].text, self.iteration
                    )
                except Exception:
                    traceback.print_exc()
            except Exception:
                traceback.print_exc()

            result[i] = full_wave if is_valid else None

        return result


    def _efficient_audio_generation(self, vo_items):
        inputs = []
        for vo_item in vo_items:
            # берем наш текст и сэмплы тембра и просодии и упаковываем в задачи для синтеза
            inputs.append(SynthesisInput(text=vo_item['text'], stress=not vo_item['is_accented'],
                                         timbre_wave_24k=vo_item['style_wave_24k'],
                                         prosody_wave_24k=vo_item['raw_wave_24k']))
        # отправляем в svr tts
        # noinspection PyUnresolvedReferences
        waves = self.synthesize_batch_with_split(inputs)

        results = []
        for idx, wave in enumerate(waves):
            # не все задачи могут быть озвучены, например из за нехватки баланса или не верного текста, их пропустим
            if wave is None:
                continue
            # собираем результаты и одновременно сопоставляем с исходной громкостью и файлом
            vo_item = vo_items[idx]
            results.append((wave, factory.svr_tts.OUTPUT_SR, vo_item['path'], vo_item['meta'], vo_item['raw_wave'], vo_item['raw_sr'],
                            vo_item['raw_wave_24k']))
        return results

    def _voice_over_items(self, vo_items):
        for vo_item in vo_items:
            # нейросеть для чисел с плавающей точкой использует 32 бита, потому приводим аудио в этот формат
            vo_item['style_wave_24k'] = vo_item['style_wave_24k'].astype(numpy.float32)
            vo_item['raw_wave_24k'] = vo_item['raw_wave_24k'].astype(numpy.float32)

        results = self._efficient_audio_generation(vo_items)
        return results

    def _voice_over_record(self, batch):
        records, batch_chars = batch
        """
            Озвучивает группу записей
        """
        vo_items = []
        for record in records:
            path = Path(f"workspace/resources/{record['audio']}")
            # Прочитаем оригинальный аудио файл и его громкость для определения просодии
            try:
                raw_wave_24k, meta, raw_wave, raw_sr = self.audio.load_audio_norm(str(path))
            except ValueError as e:
                print(e)
                continue
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
                'meta': meta,
                'style_wave_24k': style_wave_24k,
                'path': record['audio'],
                'raw_wave': raw_wave,
                'raw_sr': raw_sr
            }
            vo_items.append(vo_item)
        if not len(vo_items):
            if _progress_counter is not None and batch_chars:
                with _progress_counter.get_lock():
                    _progress_counter.value += batch_chars
            return
        # Озвучиваем батч
        results = self._voice_over_items(vo_items)

        for dub, sr, i_path, i_meta, i_raw_wave, i_raw_sr, i_raw_wave_24k in results:

            # Не значительно ускорим если это необходимо
            dub = AudioProcessor.speedup_if_need(dub, sr, i_raw_wave, i_raw_sr, self.config['max_extra_speed'])

            # Восстановим характеристики оригинального аудио
            dub, sr = self.audio.restore_meta(dub, sr, i_meta)

            # Сохраняем дубляж
            dub_file = Path(f"workspace/dub/{i_path}")
            dub_file.parent.mkdir(parents=True, exist_ok=True)
            dub_file = Path(str(dub_file)).with_suffix(f".{self.config['ext']}")

            wave_format=self.config['ext']
            codec="libvorbis" if wave_format == 'ogg' else None
            parameters=["-qscale:a", "9"] if wave_format == 'ogg' else None

            AudioProcessor.to_segment(dub, sr).export(
                dub_file,
                format=wave_format,
                codec=codec,
                parameters=parameters
            )

        # Обновляем общий прогресс по символам (один раз на батч)
        if _progress_counter is not None and batch_chars:
                with _progress_counter.get_lock():
                    _progress_counter.value += batch_chars

    def run(self):
        # Найдем все строки что нужно озвучить с учетом разных версий файлов
        todo_records, all_records = self.csv.find_changed_text_rows_csv()
        todo_records = sorted(todo_records,
                              key=lambda d: len(set(self.text.get_text(d)[0])) * len(self.text.get_text(d)[0]),
                              reverse=True)
        while len(todo_records):
            fieldnames = todo_records[0].keys()
            # Что-б снизить нагрузку на api сервера токенизации разобьем записи на небольшие на группы
            raw_batches = [todo_records[i:i + self.config['batch_size']] for i in
                           range(0, len(todo_records), self.config['batch_size'])]

            # Считаем символы по батчам в родителе, чтобы не дёргать get_text в воркерах
            batches = []
            total_chars = 0
            for b in raw_batches:
                ch = 0
                for r in b:
                    try:
                        t, _ = self.text.get_text(r)
                        ch += len(t or "")
                    except Exception:
                        pass
                batches.append((b, ch))
                total_chars += ch

            mp_context = get_context('spawn')

            progress_counter = mp_context.Value('L', 0)

            # если однопроцессный режим то надо вручную инициализировать модельки
            if self.config['n_jobs'] == 1:
                _worker_init(self.config, progress_counter)

            try:
                pqdm(batches, self._voice_over_record,
                     n_jobs=self.config['n_jobs'],
                     mp_context=mp_context,
                     smoothing=0,
                     desc='Общий прогресс',
                     total=total_chars,
                     unit='симв',
                     tqdm_class=partial(_CharSyncTqdm, progress_counter=progress_counter),
                     initializer=_worker_init,
                     initargs=(self.config, progress_counter),
                     position=0,
                     exception_behaviour='immediate',
                     dynamic_ncols=True
                     )
            except AssertionError as e:
                # Неисправимая ошибка, может кончился баланс, завершаем работу
                print(e)

            todo_records, all_records = self.csv.find_changed_text_rows_csv()
            todo_records = sorted(todo_records,
                                  key=lambda d: len(set(self.text.get_text(d)[0])) * len(self.text.get_text(d)[0]),
                                  reverse=True)

            _write_todo_csv("workspace/todo_voiceover.csv", todo_records, self.config["csv_delimiter"])

            self.iteration += 1

def _write_todo_csv(path: str, records: list[dict], delimiter: str):
    cleaned = []
    field_set = set()

    for r in records:
        rr = {k: v for k, v in r.items() if k is not None}  # убираем None-ключ
        cleaned.append(rr)
        field_set.update(rr.keys())

    fieldnames = list(field_set)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
            restval=""
        )
        w.writeheader()
        w.writerows(cleaned)