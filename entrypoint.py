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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--ext', type=str, default="wav", help='Формат результирующего аудио')
    parser.add_argument('--tone_sample_len', type=int, default=7500, help='Длина сэмпла голоса')
    parser.add_argument('--api_key', type=str, help='Ваш ключ доступа к api', required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--n_jobs', type=int, default=None, help='Количество воркеров')
    parser.add_argument('--csv_delimiter', type=str, default=',', help='Разделитель, использующийся в csv.')
    parser.add_argument('--is_respect_mos', action='store_true', default=True,
                        help='Нужно ли учитывать качество звука для построения сэмпла голоса')
    parser.add_argument('--providers', nargs='+', default=get_def_providers(),
                        help='Список провайдеров для выполнения (ONNX runtime)')
    parser.add_argument('--path_filter', type=str, default='',
                        help='Фильтр реплик попадающих в озвучку по пути файла')
    parser.add_argument('--user_models_dir', type=str, default='./workspace/model/',
                        help='Путь до кастомных моделей если есть')
    parser.add_argument('--dur_norm_low', type=float, default=5.0, help='Минимальный порог темпа речи')
    parser.add_argument('--dur_high_t0', type=float, default=1.0, help='t0 (сек) для расчёта максимального темпа')
    parser.add_argument('--dur_high_t1', type=float, default=15.0, help='t1 (сек) для расчёта максимального темпа')
    parser.add_argument('--dur_high_k', type=float, default=10.0, help='k (кривизна) для расчёта максимального темпа')
    parser.add_argument('--dur_norm_thr_low', type=float, default=0.5, help='Допустимое отклонение темпа речи от границ')
    parser.add_argument('--dur_norm_thr_high', type=float, default=4.0, help='Допустимое отклонение темпа речи от границ')
    parser.add_argument('--reinit_every', type=int, default=0, help='Очищать сессию onnx каждые n раз')
    parser.add_argument('--prosody_cond', type=float, default=0.6, help='Насколько сильно следовать оригинальной просодии')
    parser.add_argument('--min_prosody_len', type=float, default=2.0, help='Длина просодии ниже которой она свапнется на тембр')
    parser.add_argument('--max_extra_speed', type=float, default=0.15, help='Процент на сколько можно ускорить речь при необходимости')
    parser.add_argument('--vc_type', type=str, default='default', help='Тип конверсии голоса')
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = detect_gpu_count()
    config = vars(args)
    PipelineModule(config).run()
