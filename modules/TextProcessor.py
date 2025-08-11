import difflib
import re


class TextProcessor:
    def __init__(self):
        pass

    def get_text(self, record):
        """
            Берет текст из csv файла в порядке
            1 - Если есть вариант текста с вручную установленными ударениями то бере его
            2 - Если есть адаптированный под длину аудио текст то берем его
            3 - Берем текст
        """
        if 'text_accented' in record.keys() and record['text_accented'].strip():
            return record['text_accented'], True
        elif 'text_adopted' in record.keys() and record['text_adopted'].strip():
            return record['text_adopted'], False
        else:
            return record['text'], False

    def is_sound_word(self, sentence):
        if sentence and 'да' in sentence.lower():
            return False
        # Регулярное выражение для звуковых слов, учитывающее повторяющиеся и чередующиеся буквы
        sound_pattern = re.compile(r'^(?:[А-Яа-я]{1,3}(?:[-–—][А-Яа-я]{1,3})+)[!?.…]*$')

        # Убираем лишние пробелы и проверяем предложение
        sentence = sentence.strip()
        result = bool(sound_pattern.match(sentence))
        if result:
            print('sound word ' + sentence)
        return result

    def split_text(self, text, max_text_len):
        phrases = re.split(r'(?<=[.!?…])\s+', text)
        chunks, current = [], ""
        for phrase in phrases:
            if len(current) + len(phrase) + 1 <= max_text_len:
                current += (" " if current else "") + phrase
            else:
                if current:
                    chunks.append(current)
                current = phrase
        if current:
            chunks.append(current)
        return chunks


def normalize(text):
    text = text.lower().replace('ё', 'е').replace('й', 'и')
    # нижний регистр, убрать пунктуацию, пробелы
    text = re.sub(r'[^\w]', '', text)
    # удалить повторяющиеся подряд символы (например: "оооо" → "о")
    text = re.sub(r'(.)\1+', r'\1', text)
    return text


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()
