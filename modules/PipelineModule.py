import csv
import threading
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy
import onnxruntime
from pqdm.processes import pqdm
from svr_tts.core import SynthesisInput
from tqdm.auto import tqdm

from modules.AudioProcessor import AudioProcessor
from modules.CsvProcessor import CsvProcessor
from modules.DedupCsv import DedupCsv
from modules.ModelFactory import ModelFactory
from modules.SpeakerProcessor import SpeakerProcessor
from modules.TextProcessor import TextProcessor

INPUT_SR = 24_000
OUTPUT_SR = 22_050

FADE_LEN = int(0.1 * OUTPUT_SR)

onnxruntime.set_default_logger_severity(3)
factory = None
_voiced_chars_counter = None


def _worker_init(config, voiced_chars_counter=None):
    global factory, _voiced_chars_counter
    factory = ModelFactory(config)
    _voiced_chars_counter = voiced_chars_counter


def _add_shared_value(counter, value):
    if counter is None or not value:
        return
    with counter.get_lock():
        counter.value += value


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

        def _run(batch_inputs):
            return factory.svr_tts.synthesize(
                batch_inputs,
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

        try:
            all_waves = _run(inputs)

        except Exception:
            traceback.print_exc()

            # fallback: пытаемся по одному, чтобы не терять весь батч
            all_waves = [None] * len(inputs)
            for i, inp in enumerate(inputs):
                try:
                    one = _run([inp])
                    all_waves[i] = one[0] if one else None
                except Exception:
                    traceback.print_exc()
                    all_waves[i] = None

        result = [None] * len(inputs)

        for i, full_wave in enumerate(all_waves):
            if full_wave is None:
                continue
            try:
                is_valid = self.audio.validate(full_wave, factory.svr_tts.OUTPUT_SR, inputs[i].text, self.iteration)
            except Exception:
                traceback.print_exc()
                is_valid = False
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
            results.append((wave, factory.svr_tts.OUTPUT_SR, vo_item['path'], vo_item['meta'], vo_item['raw_wave'],
                            vo_item['raw_sr'],
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
        records, _ = batch
        """
            Озвучивает группу записей
        """

        dedup = DedupCsv(self.config.get('dedup_csv', 'workspace/dedup.csv'))
        dedup_map = dedup.load_map()

        # дедуп внутри батча (чтобы не отправлять одинаковые задачи в TTS)
        batch_seen = {}  # sha -> canonical audio path (record['audio'])
        batch_dups = {}  # sha -> [(dup audio path, text_len)]
        sha_by_path = {}  # canonical audio path -> sha

        vo_items = []
        voiced_chars = 0
        for record in records:
            path = Path(f"workspace/resources/{record['audio']}")
            # Определяем текст для озвучки

            text, is_accented = self.text.get_text(record)
            text_len = len(text or "")
            try:
                sha = DedupCsv.sha256(path, text)
            except Exception:
                sha = None

            if sha:
                # 1) если уже озвучено ранее — просто копируем и идём дальше
                voiced_path = dedup_map.get(sha)
                if voiced_path and Path(voiced_path).exists():
                    dub_file = Path(f"workspace/dub/{record['audio']}")
                    dub_file.parent.mkdir(parents=True, exist_ok=True)
                    dub_file = Path(str(dub_file)).with_suffix(f".{self.config['ext']}")
                    try:
                        DedupCsv.link_or_copy(voiced_path, dub_file)
                        voiced_chars += text_len
                    except Exception:
                        traceback.print_exc()
                    continue

                # 2) дедуп внутри текущего батча — не добавляем второй раз в vo_items
                if sha in batch_seen:
                    batch_dups.setdefault(sha, []).append((record['audio'], text_len))
                    continue

                batch_seen[sha] = record['audio']
                sha_by_path[record['audio']] = sha

            # Прочитаем оригинальный аудио файл и его громкость для определения просодии
            try:
                raw_wave_24k, meta, raw_wave, raw_sr = self.audio.load_audio_norm(str(path))
            except ValueError as e:
                print(e)
                continue
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
                'sha256': sha,
                'text_len': text_len,
                'raw_wave': raw_wave,
                'raw_sr': raw_sr
            }
            vo_items.append(vo_item)
        if not len(vo_items):
            _add_shared_value(_voiced_chars_counter, voiced_chars)
            return
        # Озвучиваем батч
        results = self._voice_over_items(vo_items)
        success_paths = set()
        text_len_by_path = {item['path']: item['text_len'] for item in vo_items}

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
            success_paths.add(i_path)
            voiced_chars += text_len_by_path.get(i_path, 0)

            # sha для этого результата (быстро через sha_by_path; fallback — как раньше)
            sha = sha_by_path.get(i_path)
            if sha is None:
                try:
                    for it in vo_items:
                        if it.get('path') == i_path:
                            sha = it.get('sha256')
                            break
                except Exception:
                    sha = None

            # дописываем в dedup.csv
            try:
                if sha:
                    dedup.append(sha, str(dub_file))
            except Exception:
                traceback.print_exc()

            # размножаем результат на дубли внутри батча
            if sha and sha in batch_dups:
                for dup_audio, dup_text_len in batch_dups.get(sha, []):
                    try:
                        dup_file = Path(f"workspace/dub/{dup_audio}")
                        dup_file.parent.mkdir(parents=True, exist_ok=True)
                        dup_file = Path(str(dup_file)).with_suffix(f".{self.config['ext']}")
                        DedupCsv.link_or_copy(dub_file, dup_file)
                        voiced_chars += dup_text_len
                    except Exception:
                        traceback.print_exc()

        _add_shared_value(_voiced_chars_counter, voiced_chars)

    def run(self):
        # Найдем все строки что нужно озвучить с учетом разных версий файлов
        todo_records, all_records = self.csv.find_changed_text_rows_csv()
        all_changed_records = self.csv.find_all_changed_text_rows_csv()
        todo_records = sorted(todo_records,
                              key=lambda d: len(set(self.text.get_text(d)[0])) * len(self.text.get_text(d)[0]),
                              reverse=True)
        todo_audio = {record['audio'] for record in todo_records}
        total_chars = 0
        completed_chars = 0
        for record in all_changed_records:
            try:
                text, _ = self.text.get_text(record)
                text_len = len(text or "")
                total_chars += text_len
                if record['audio'] not in todo_audio:
                    completed_chars += text_len
            except Exception:
                pass

        mp_context = get_context('spawn')
        voiced_chars_counter = mp_context.Value('L', completed_chars)
        max_validation_passes = self.audio.get_validation_pass_count()

        # если однопроцессный режим то надо вручную инициализировать модельки
        if self.config['n_jobs'] == 1:
            _worker_init(self.config, voiced_chars_counter)

        stop_evt = threading.Event()

        def _progress_desc():
            current_pass = min(self.iteration + 1, max_validation_passes)
            return f'Общий прогресс [{current_pass}/{max_validation_passes}]'

        def _watch_chars():
            last = completed_chars
            shown_pass = None
            with tqdm(total=total_chars, initial=completed_chars, desc=_progress_desc(), unit='симв', smoothing=0,
                      dynamic_ncols=True, position=0, leave=True) as pbar:
                while not stop_evt.is_set():
                    current_pass = min(self.iteration + 1, max_validation_passes)
                    if current_pass != shown_pass:
                        pbar.set_description(_progress_desc(), refresh=False)
                        shown_pass = current_pass
                    cur = voiced_chars_counter.value
                    if cur > last:
                        pbar.update(cur - last)
                        last = cur
                    time.sleep(0.2)
                # финальный добор
                cur = voiced_chars_counter.value
                if cur > last:
                    pbar.update(cur - last)

        t = None
        if total_chars:
            t = threading.Thread(target=_watch_chars, daemon=True)
            t.start()

        try:
            while len(todo_records):
                # Что-б снизить нагрузку на api сервера токенизации разобьем записи на небольшие на группы
                raw_batches = [todo_records[i:i + self.config['batch_size']] for i in
                               range(0, len(todo_records), self.config['batch_size'])]

                # Считаем символы по батчам в родителе (для отдельного прогресс-бара)
                batches = []
                for b in raw_batches:
                    ch = 0
                    for r in b:
                        try:
                            t_text, _ = self.text.get_text(r)
                            ch += len(t_text or "")
                        except Exception:
                            pass
                    batches.append((b, ch))

                pqdm(batches, self._voice_over_record,
                     n_jobs=self.config['n_jobs'],
                     mp_context=mp_context,
                     smoothing=0,
                     desc='Общий прогресс (батчи)',
                     initializer=_worker_init,
                     initargs=(self.config, voiced_chars_counter),
                     position=1,
                     exception_behaviour='immediate',
                     dynamic_ncols=True,
                     disable=True
                )

                todo_records, all_records = self.csv.find_changed_text_rows_csv()
                todo_records = sorted(todo_records,
                                      key=lambda d: len(set(self.text.get_text(d)[0])) * len(self.text.get_text(d)[0]),
                                      reverse=True)

                _write_todo_csv("workspace/todo_voiceover.csv", todo_records, self.config["csv_delimiter"])

                self.iteration += 1
        except AssertionError as e:
            # Неисправимая ошибка, может кончился баланс, завершаем работу
            print(e)
        finally:
            stop_evt.set()
            if t is not None:
                t.join()

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