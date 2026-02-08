import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendors" / "CosyVoice"))
import argparse

import GPUtil

from modules.PipelineModule import PipelineModule


def detect_gpu_count():
    gpus = GPUtil.getGPUs()
    return len(gpus) if gpus else 1


def get_def_providers():
    if len(GPUtil.getGPUs()):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"bool expected, got: {v}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--ext', type=str, default="wav", help='Формат результирующего аудио')
    parser.add_argument('--tone_sample_len', type=int, default=10000, help='Длина сэмпла голоса')
    parser.add_argument('--api_key', type=str, help='Ваш ключ доступа к api', required=True)
    parser.add_argument('--batch_size', type=int, default=24, help='Размер батча')
    parser.add_argument('--n_jobs', type=int, default=None, help='Количество воркеров')
    parser.add_argument('--csv_delimiter', type=str, default=',', help='Разделитель, использующийся в csv.')
    # По умолчанию MOS учитывается. Если нужно отключить — используйте --no_respect_mos.
    # (Оставляем --is_respect_mos для обратной совместимости со старыми скриптами.)
    parser.add_argument('--is_respect_mos', action='store_true', default=True,
                        help='Учитывать качество звука (MOS) при построении сэмпла голоса')
    parser.add_argument('--no_respect_mos', action='store_false', dest='is_respect_mos',
                        help='Не учитывать качество звука (MOS) при построении сэмпла голоса')
    parser.add_argument('--providers', nargs='+', default=get_def_providers(),
                        help='Список провайдеров для выполнения (ONNX runtime)')
    parser.add_argument('--path_filter', type=str, default='',
                        help='Фильтр реплик попадающих в озвучку по пути файла')
    parser.add_argument('--user_models_dir', type=str, default='./workspace/model/',
                        help='Путь до кастомных моделей если есть')
    parser.add_argument('--reinit_every', type=int, default=0, help='Очищать сессию onnx каждые n раз')
    parser.add_argument('--prosody_cond', type=float, default=0.6, help='Насколько сильно следовать оригинальной просодии')
    parser.add_argument('--min_prosody_len', type=float, default=2.0, help='Длина просодии ниже которой она свапнется на тембр')
    parser.add_argument('--speed_search_attempts', type=int, default=6, help='Количество попыток уточнения скорости (автоподбор)')
    parser.add_argument('--speed_adjust_step_pct', type=float, default=0.08, help='Шаг уточнения скорости Y (0.08 = 8%)')
    parser.add_argument('--speed_clip_min', type=float, default=0.5, help='Минимальная скорость для клиппинга')
    parser.add_argument('--speed_clip_max', type=float, default=2.0, help='Максимальная скорость для клиппинга')

    parser.add_argument('--max_extra_speed', type=float, default=0.15, help='Процент на сколько можно ускорить речь при необходимости')

    parser.add_argument('--len_t_short', type=float, default=1.0, help='Реплики короче этого (сек) считаем короткими.')
    parser.add_argument('--len_t_long', type=float, default=15.0, help='Реплики длиннее этого (сек) считаем длинными.')

    parser.add_argument('--max_longer_pct_short', type=float, default=0.15, help='Для коротких: сколько максимум можно быть длиннее семпла (0.35 = 35%).')
    parser.add_argument('--max_longer_pct_long', type=float, default=0.05, help='Для длинных: сколько максимум можно быть длиннее семпла (0.15 = 15%).')

    parser.add_argument('--max_shorter_pct_short', type=float, default=0.15, help='Для коротких: сколько максимум можно быть короче семпла (0.25 = 25%).')
    parser.add_argument('--max_shorter_pct_long', type=float, default=0.05, help='Для длинных: сколько максимум можно быть короче семпла (0.10 = 10%).')

    parser.add_argument('--vc_type', type=str, default='default', help='Тип конверсии голоса')
    parser.add_argument('--put_yo', type=str2bool, required=True,
                        help='Ставить букву ё в тексте (true/false)')
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = detect_gpu_count()
    config = vars(args)
    PipelineModule(config).run()
