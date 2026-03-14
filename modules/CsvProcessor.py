import csv
import re
from functools import lru_cache
from glob import glob
from pathlib import Path


class CsvProcessor:
    def __init__(self, config, text_processor):
        self.ext = config['ext']
        self.text = text_processor
        self.csv_delimiter = config['csv_delimiter']
        self.path_filter = None
        if config['path_filter']:
            self.path_filter = config['path_filter'].replace('\\', '/').removeprefix('./').removeprefix('/')
        pass

    @staticmethod
    def _normalize_audio_path(audio):
        raw_audio = str(audio or '').strip().replace('\\', '/')
        if not raw_audio or raw_audio == '.':
            return ''
        return str(Path(raw_audio)).replace('\\', '/').removeprefix('./')

    def _audio_key(self, audio):
        normalized_audio = self._normalize_audio_path(audio)
        if not normalized_audio:
            return ''

        path = Path(normalized_audio)
        if not path.name:
            return ''

        return f"{path.parent}\\{path.stem}".replace('\\', '/').removeprefix('./').lower()

    def _extract_version_number(self, filename):
        # Получаем имя файла без пути и расширения
        name = Path(filename).stem

        # Ищем последнюю цифру в имени файла
        match = re.search(r'(\d+)$', name)
        if match:
            return int(match.group(1))
        else:
            # Если цифра не найдена, возвращаем 0
            return 0

    def _collect_existing_sets(self):
        resources = set(
            map(lambda i: f"{Path(i).parent}\\{Path(i).stem}".replace('\\', '/').removeprefix('./').lower(),
                glob('**/*.*', recursive=True, root_dir='workspace/resources')))
        dub = set(
            map(lambda i: f"{Path(i).parent}\\{Path(i).stem}".replace('\\', '/').removeprefix('./').lower(),
                glob('**/*.*', recursive=True, root_dir='workspace/dub')))
        return resources, dub

    def _filter_by_exists(self, records, all_records):
        resources, dub = self._collect_existing_sets()

        todo = set(resources) - set(dub)

        records_with_text = [
            row for row in records
            if row.get('audio', '').strip()
               and re.search(r'[А-Яа-яЁё]', self.text.get_text(row)[0])
        ]
        print(f"Найдено {len(records_with_text)} записей содержащих аудио и русский текст из {len(records)}")
        records_with_files = [row for row in records_with_text if self._audio_key(row.get('audio')) in resources]
        if len(records_with_files) != len(records_with_text):
            print('Внимание в csv есть файлы которых нет а рабочей директории')
            nf = [r for r in records_with_text if self._audio_key(r.get('audio')) not in resources]
            sel = nf[:5] + [r for r in nf[-5:] if r not in nf[:5]]

            print(f"Всего не найдено: {len(nf)}")
            for i, r in enumerate(sel, 1):
                print(f"Пример: {i}) {r['audio']}")
        todo_records_with_files = [row for row in records_with_text if self._audio_key(row.get('audio')) in todo]
        records_with_text = todo_records_with_files
        all_records = [row for row in all_records if
                       row.get('audio', '').strip()
                       and re.search(r'[А-Яа-яЁё]', self.text.get_text(row)[0])
                       and self._audio_key(row.get('audio')) in resources]
        records_with_text = sorted(records_with_text, key=lambda i: i['audio'])
        return records_with_text, all_records

    def _find_changed_rows_csv(self):
        """
        Возвращает измененные строки и все актуальные строки из набора версий без учета наличия dub.
        """
        all_version_files = glob("workspace/voiceover**.csv")

        # Проверяем, найдены ли файлы
        if not all_version_files:
            print("Файлы не найдены по заданному шаблону.")
            return [], []

        # Сортируем файлы по номеру версии
        all_version_files.sort(key=lambda x: self._extract_version_number(x))

        # Последняя версия - файл с самым большим номером версии
        latest_version = all_version_files[-1]
        previous_versions = all_version_files[:-1]

        print(f"Найдены версии файлов: {all_version_files}")
        print(f"Последняя версия: {latest_version}")
        print(f"Предыдущие версии: {previous_versions}")

        # Словарь для хранения предыдущих значений text_column по ключу key_column
        previous_data = {}

        # Читаем и обрабатываем предыдущие версии
        for file in previous_versions:
            with open(file, mode='r', encoding='utf-8-sig', newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=self.csv_delimiter)
                fieldnames = reader.fieldnames

                # Проверяем наличие столбца 'audio'
                if 'audio' not in fieldnames:
                    print(f"Столбец 'audio' не найден в файле {file}.")
                    continue  # Пропускаем этот файл

                for row in reader:
                    row['audio'] = self._normalize_audio_path(row.get('audio'))
                    if not row['audio']:
                        continue
                    if self.path_filter and not row['audio'].startswith(self.path_filter):
                        continue
                    key = row['audio']
                    previous_data[key] = row

        # Список для хранения измененных строк
        todo_rows = []

        # Читаем и обрабатываем последнюю версию
        with open(latest_version, mode='r', encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=self.csv_delimiter)
            fieldnames = reader.fieldnames

            # Проверяем наличие столбца 'audio'
            if 'audio' not in fieldnames:
                print(f"Столбец 'audio' не найден в файле {latest_version}.")
                return [], []

            for row in reader:
                row['audio'] = self._normalize_audio_path(row.get('audio'))
                if not row['audio']:
                    continue
                if self.path_filter and not row['audio'].startswith(self.path_filter):
                    continue
                key = row['audio']
                text = self.text.get_text(row)[0]

                if key in previous_data.keys():
                    # Если текст изменился, добавляем строку в список измененных
                    if text != self.text.get_text(previous_data[key])[0] or row['speaker'] != previous_data[key]['speaker']:
                        todo_rows.append(row)
                        previous_data[key] = row
                else:
                    # Добавляем новые строки в список
                    todo_rows.append(row)
                    previous_data[key] = row

        return todo_rows, list(previous_data.values())

    def find_changed_text_rows_csv(self):
        """
        Находит csv файлы, свравнивает их содержимое и находит изменившиеся строки.
        :return:
        """
        todo_rows, all_rows = self._find_changed_rows_csv()
        todo_records, all_records = self._filter_by_exists(todo_rows, all_rows)
        todo_records = list(filter(lambda i: not self.text.is_sound_word(self.text.get_text(i)[0]), todo_records))
        all_records = list(filter(lambda i: not self.text.is_sound_word(self.text.get_text(i)[0]), all_records))
        return todo_records, all_records

    def find_all_changed_text_rows_csv(self):
        """
        Возвращает все измененные строки, даже если для них уже есть dub-файл.
        """
        changed_rows, _ = self._find_changed_rows_csv()
        resources, _ = self._collect_existing_sets()

        records_with_text = [
            row for row in changed_rows
            if row.get('audio', '').strip() and re.search(r'[А-Яа-яЁё]', self.text.get_text(row)[0])
        ]
        records_with_files = [row for row in records_with_text if self._audio_key(row.get('audio')) in resources]
        records_with_files = list(filter(lambda i: not self.text.is_sound_word(self.text.get_text(i)[0]), records_with_files))
        return sorted(records_with_files, key=lambda i: i['audio'])

    @lru_cache
    def load_stress_exclusions(self, path='workspace/stress_dict.csv') -> dict[str, str]:
        """
        Читает CSV вида: исходное_слово,замена
        пример: детектив,дэтэкИв
        Если файла нет — возвращает {}.
        """
        p = Path(path)
        if not p.is_file():
            return {}

        mapping: dict[str, str] = {}

        try:
            with p.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 2:
                        continue

                    src = row[0].strip()
                    dst = row[1].strip()

                    if not src or not dst:
                        continue

                    mapping[src] = dst
        except OSError:
            # на случай проблем с чтением файла
            return {}

        return mapping
