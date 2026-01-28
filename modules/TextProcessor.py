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

    def is_sound_word(self, sentence, stop_words=None):
        inclusions = {
            'угу', 'а', 'эй', 'ох', 'ой', 'хм', 'ммм', 'а-а', 'ааа', 'ха', 'фу', 'бах', 'бдыщ', 'бум', 'о', 'упс',
            'пфе', 'ай', 'хммм'
        }

        if stop_words is None:
            stop_words = {'да', 'нет', 'ну', 'ага', 'так', 'ну-ка', 'спи', 'вот', 'не-а'}

        if not sentence:
            return False

        s_low = sentence.strip().lower()
        s_low = re.sub(r'[–—]', '-', s_low)
        s_norm = re.sub(r'^[\s\W_]+|[\s\W_]+$', '', s_low)
        if s_norm in inclusions:
            return True

        s_low2 = sentence.lower()
        if any(w in s_low2 for w in stop_words):
            return False

        sound_pattern = re.compile(r'^(?:[А-Яа-я]{1,3}(?:[-–—][А-Яа-я]{1,3})+)[!?.…]*$')

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

def _allowed_l_errors(g_l: int, threshold: float, base_thr: float = 0.75) -> int:
    # при 0.75 -> extra=0
    # 0.70 -> +1, 0.65 -> +2, 0.60 -> +3 ...
    extra = int(max(0.0, (base_thr - threshold) / 0.05))
    extra = min(extra, 4)  # кап, подстрой если надо

    base = g_l // 4  # 1..3->0, 4..7->1, 8..11->2 ...
    return base + extra

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

    allowed = _allowed_l_errors(t_l, threshold)
    ok_l = abs(t_l - g_l) <= allowed
    # if not ok_l:
    #     print(f"ok_l {abs(t_l - g_l)} allowed = {allowed}")

    t_words = re.findall(r"\w+", t, flags=re.UNICODE)
    g_words = re.findall(r"\w+", g, flags=re.UNICODE)

    t_last = t_words[-1] if t_words else ""
    t_first = t_words[0] if t_words else ""
    g_last = g_words[-1] if g_words else ""
    g_first = g_words[0] if g_words else ""

    sim_last = similarity(t_last, g_last)
    # if not sim_last >= threshold:
    #     print(f"sim_last = {t_last} target: {g_last}")
    sim_first = similarity(t_first, g_first)
    # if not sim_first >= threshold:
    #     print(f"sim_first = {t_first} target {g_first}")

    is_sim = sim_last >= threshold and sim_first >= threshold

    ok = ok_l and is_sim

    return not ok
