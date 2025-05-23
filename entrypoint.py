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

    parser.add_argument('--ext', type=str, default="ogg", help='Формат результирующего ауидо')
    parser.add_argument('--tone_sample_len', type=int, default=15000, help='Длина сэмпла голоса')
    parser.add_argument('--api_key', type=str, help='Ваш ключ доступа к api', required=True)
    parser.add_argument('--batch_size', type=int, default=20, help='Размер батча')
    parser.add_argument('--n_jobs', type=int, default=None, help='Количество воркеров')
    parser.add_argument('--csv_delimiter', type=str, default=',', help='Разделитель, использующийся в csv.')
    parser.add_argument('--is_strict_len', type=bool, default=False,
                        help='Должна ли длина аудио строго совпадать с оригиналом')
    parser.add_argument('--is_respect_mos', type=bool, default=False,
                        help='Нужно ли учитывать качество звука для построения сэмпла голоса')
    parser.add_argument('--providers', nargs='+', default=get_def_providers(),
                        help='Список провайдеров для выполнения (ONNX runtime)')
    parser.add_argument('--path_filter', type=str, default=None,
                        help='Фильтр реплик попадающих в озвучку по пути файла')
    parser.add_argument('--min_len_deviation', type=int, default=0.75,
                        help='Минимальный порог длины синтезированной волны для повторного синтеза в случае галюцинации')
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = detect_gpu_count()

    config = vars(args)
    PipelineModule(config).run()
