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

    parser.add_argument('--ext', type=str, default="wav", help='Формат результирующего ауидо')
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
    parser.add_argument('--user_models_dir', type=str, default='',
                        help='Путь до кастомных моделей если есть')
    parser.add_argument('--dur_norm_low', type=float, default=5.0, help='Минимальный порог темпа речи')
    parser.add_argument('--dur_norm_high', type=float, default=16.0, help='Максимальный порог темпа речи')
    parser.add_argument('--dur_norm_thr_low', type=float, default=0.5, help='Допустимое отклонение темпа речи от границ')
    parser.add_argument('--dur_norm_thr_high', type=float, default=5.0, help='Допустимое отклонение темпа речи от границ')
    parser.add_argument('--reinit_every', type=int, default=32, help='Очищать сессию onnx каждые n раз')
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = detect_gpu_count()
    config = vars(args)
    PipelineModule(config).run()
