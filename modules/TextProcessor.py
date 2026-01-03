import difflib
import re
from typing import List


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

    def split_text(self, text: str, max_text_len: int, splitter: str = r'(?<=[.!?…])\s+') -> List[str]:
        phrases = re.split(splitter, text.strip()) if text else []
        chunks, cur = [], ""
        for ph in phrases:
            if not ph:
                continue
            add = ((" " if cur else "") + ph)
            if len(cur) + len(add) <= max_text_len:
                cur += add
            else:
                if cur:
                    chunks.append(cur)
                cur = ph
        if cur:
            chunks.append(cur)
        return chunks

def normalize(text, remove_spaces=True):
    text = text.lower().replace('ё', 'е').replace('й', 'и')
    # нижний регистр, убрать пунктуацию, пробелы
    if remove_spaces:
        text = re.sub(r'[^\w]', '', text)
    # удалить повторяющиеся подряд символы (например: "оооо" → "о")
    text = re.sub(r'(.)\1+', r'\1', text)
    return text


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def is_tts_hallucination(text: str, target: str, threshold=0.75) -> bool:
    """
    True = похоже на галлюцинацию (не совпало кол-во 'ль' ИЛИ не совпало последнее слово)
    False = всё ок по этим двум проверкам
    """
    t = normalize(text, False)
    g = normalize(target, False)

    # 1) совпадение количества 'ль'
    t_l = t.count("ль")
    g_l = g.count("ль")

    # 2) совпадение последнего слова
    # берём только слова (буквы/цифры/подчёркивание) — знаки препинания отваливаются
    t_words = re.findall(r"\w+", t, flags=re.UNICODE)
    g_words = re.findall(r"\w+", g, flags=re.UNICODE)

    t_last = t_words[-1] if t_words else ""
    g_last = g_words[-1] if g_words else ""

    sim = similarity(t_last, g_last)
    is_sim = sim >= threshold

    if threshold >= 0.6:
        ok = (t_l == g_l) and is_sim
    else:
        ok = is_sim

    return not ok
