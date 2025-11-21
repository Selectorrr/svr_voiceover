import csv
import sys
from collections import Counter

from entrypoint import get_def_providers
from modules.CsvProcessor import CsvProcessor
from modules.ModelFactory import ModelFactory
from modules.TextProcessor import TextProcessor

config = {
    'api_key': sys.argv[2].split('=')[1],
    'csv_delimiter': ',',
    'ext': 'ogg',
    'path_filter': '',
    'tone_sample_len': 7500,
    'is_respect_mos': False,
    'is_strict_len': True,
    'is_use_voice_len': True,
    'providers': get_def_providers(),
    'user_models_dir': ''
}
text_module = TextProcessor()
csv_module = CsvProcessor(config, text_module)

factory = None


def _worker_init(config):
    global factory
    factory = ModelFactory(config)


_worker_init(config)

from collections import defaultdict

def build_balanced_subset(items, coverage: float = 0.85, target_total: int = 10_000):
    """
    1) Считаем counts[sp].
    2) Отбираем ядро спикеров, которые дают coverage (0.85) объёма реплик.
    3) В ядре:
        - min_c = мин(count)
        - scale ≈ target_total * min_c / core_total
        - ideal_sp = (count_sp / min_c) * scale
        - k_sp ≈ ideal_sp, с корректировкой так, чтобы суммарно было target_total.
        - берём k_sp самых длинных реплик по text.
    """
    if not items or target_total <= 0:
        return []

    # --- группировка по speaker ---
    by_speaker = defaultdict(list)
    for d in items:
        sp = d.get("speaker")
        if sp is None:
            continue
        by_speaker[sp].append(d)

    if not by_speaker:
        return []

    counts = {sp: len(lst) for sp, lst in by_speaker.items()}
    total = sum(counts.values())
    if total == 0:
        return []

    coverage = max(0.0, min(1.0, float(coverage)))

    # --- шаг 1–2: ядро по кумулятивному покрытию ---
    speakers_sorted = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    core_speakers = []
    cum = 0
    for sp, c in speakers_sorted:
        core_speakers.append(sp)
        cum += c
        if cum / total >= coverage:
            break

    if not core_speakers:
        return []

    counts_core = {sp: counts[sp] for sp in core_speakers}
    core_total = sum(counts_core.values())
    min_c = min(counts_core.values())
    if min_c <= 0:
        min_c = 1

    # если target >= объёма ядра — можно просто вернуть всё ядро
    if target_total >= core_total:
        result = []
        for sp in core_speakers:
            result.extend(by_speaker[sp])
        return result

    # --- шаг 3: прямой расчёт scale и k_sp ---
    # scale так, чтобы ожидаемая сумма была целевой
    # (если игнорировать клип и округление)
    scale = target_total * min_c / core_total

    speakers = core_speakers

    ideal = {}
    base = {}
    frac = {}
    sum_base = 0

    for sp in speakers:
        c = counts_core[sp]
        ratio = c / min_c
        x = ratio * scale
        k = int(x)  # floor

        if k < 1:
            k = 1
        if k > c:
            k = c

        ideal[sp] = x
        base[sp] = k
        frac[sp] = x - k  # дробная часть
        sum_base += k

    diff = target_total - sum_base

    # --- корректировка: добрасываем / убираем по 1, ориентируясь на дробные части ---
    if diff > 0:
        # надо добавить diff штук: берём тех, у кого дробная часть больше,
        # и у кого ещё есть запас до исходного c
        candidates = sorted(
            speakers,
            key=lambda sp: frac[sp],
            reverse=True
        )
        i = 0
        while diff > 0 and i < len(candidates):
            sp = candidates[i]
            if base[sp] < counts_core[sp]:
                base[sp] += 1
                diff -= 1
            else:
                i += 1

    elif diff < 0:
        need = -diff
        # надо убрать: идём от наименьших дробных частей
        candidates = sorted(
            speakers,
            key=lambda sp: frac[sp]
        )
        i = 0
        while need > 0 and i < len(candidates):
            sp = candidates[i]
            if base[sp] > 1:
                base[sp] -= 1
                need -= 1
            else:
                i += 1
        # если need > 0 — значит target_total < кол-ва спикеров, но
        # по-хорошему до этого лучше не доводить

    # --- собираем итог ---
    result = []
    for sp in speakers:
        keep_n = base[sp]
        grp_sorted = sorted(
            by_speaker[sp],
            key=lambda x: len(x.get("text", "")),
            reverse=True,
        )
        result.extend(grp_sorted[:keep_n])

    # финальная подстраховка
    if len(result) > target_total:
        result = result[:target_total]

    return result


def save_voice_index_csv(path: str, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in records:
            audio = r.get("audio")
            if audio is not None:
                writer.writerow([audio])


def _collect_counts(items):
    c = Counter()
    for d in items:
        sp = d.get("speaker")
        if sp is not None:
            c[sp] += 1
    return c


def print_stats(title: str, items):
    counts = _collect_counts(items)
    if not counts:
        print(f"=== {title}: пусто ===")
        return

    total = sum(counts.values())
    num_speakers = len(counts)
    min_c = min(counts.values())
    max_c = max(counts.values())

    print(f"=== {title} ===")
    print(f"Строк: {total}")
    print(f"Спикеров: {num_speakers}")
    print(f"Мин. реплик на спикера: {min_c}")
    print(f"Макс. реплик на спикера: {max_c}")

    print("Топ 10 спикеров:")
    for sp, cnt in counts.most_common(10):
        pct = cnt * 100.0 / total
        print(f"  {sp}: {cnt} ({pct:.2f}%)")

    # сколько редких с мин. количеством
    rare_count = sum(1 for v in counts.values() if v == min_c)
    print(f"Спикеров с минимальным числом реплик ({min_c}): {rare_count}")
    print()


def main():
    _, all_records = csv_module.find_changed_text_rows_csv()

    print_stats("До ужатия", all_records)

    balanced = build_balanced_subset(all_records, coverage=0.85, target_total=10_000)

    print_stats("После ужатия", balanced)

    print(f"Всего строк исходно: {len(all_records)}")
    print(f"После ужатия: {len(balanced)}")

    save_voice_index_csv("workspace/voice_index.csv", balanced)


if __name__ == '__main__':
    main()
