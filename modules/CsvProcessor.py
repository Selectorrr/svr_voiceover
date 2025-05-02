import csv
import re
from glob import glob
from pathlib import Path


class CsvProcessor:
    def __init__(self, config, text_processor):
        self.ext = config['ext']
        self.text = text_processor
        self.csv_delimiter = config['csv_delimiter']
        pass

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

    def _filter_by_exists(self, records, all_records):
        all_wavs = set(
            map(lambda i: f"{Path(i).parent}\\{Path(i).stem}".replace('\\', '/').removeprefix('./'),
                glob('**/*.*', recursive=True, root_dir='workspace/resources')))
        done_wavs = set(
            map(lambda i: f"{Path(i).parent}\\{Path(i).stem}".replace('\\', '/').removeprefix('./'),
                glob('**/*.*', recursive=True, root_dir='workspace/dub')))

        todo = set(all_wavs) - set(done_wavs)

        todo_records = [row for row in records if
                        re.search(r'[А-Яа-яЁё]', self.text.get_text(row)[0]) and row['audio'].rsplit('.', 1)[0] in todo]
        all_records = [row for row in all_records if
                       re.search(r'[А-Яа-яЁё]', self.text.get_text(row)[0]) and row['audio'].rsplit('.', 1)[0] in all_wavs]
        todo_records = sorted(todo_records, key=lambda i: i['audio'])
        return todo_records, all_records

    def find_changed_text_rows_csv(self):
        """
        Находит csv файлы, свравнивает их содержимое и находит изменившиеся строки.
        :return:
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
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames

                # Проверяем наличие столбца 'audio'
                if 'audio' not in fieldnames:
                    print(f"Столбец 'audio' не найден в файле {file}.")
                    continue  # Пропускаем этот файл

                for row in reader:
                    row['audio'] = str(Path(row['audio'])).replace('\\', '/').removeprefix('./')
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
                row['audio'] = str(Path(row['audio'])).replace('\\', '/').removeprefix('./')
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

        todo_records, all_records = self._filter_by_exists(todo_rows, previous_data.values())
        todo_records = list(filter(lambda i: not self.text.is_sound_word(self.text.get_text(i)[0]), todo_records))
        all_records = list(filter(lambda i: not self.text.is_sound_word(self.text.get_text(i)[0]), all_records))
        return todo_records, all_records
