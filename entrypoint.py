import argparse

import GPUtil

from modules.PipelineModule import PipelineModule


def detect_gpu_count():
    gpus = GPUtil.getGPUs()
    return len(gpus) if gpus else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--ext', type=str, default="ogg", help='Формат результирующего ауидо')
    parser.add_argument('--tone_sample_len', type=int, default=15000, help='Длина сэмпла голоса')
    parser.add_argument('--api_key', type=str, help='Ваш ключ доступа к api', required=True)
    parser.add_argument('--batch_size', type=int, default=20, help='Размер батча')
    parser.add_argument('--n_jobs', type=int, default=None, help='Количество воркеров')
    parser.add_argument('--max_speed_ratio', type=int, default=1.15, help='На сколько можно ускорить аудио после синтеза')
    parser.add_argument('--is_strict_len', type=bool, default=False,
                        help='Должна ли длина аудио строго совпадать с оригиналом')
    parser.add_argument('--providers', nargs='+',
                        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        help='Список провайдеров для выполнения (ONNX runtime)')
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = detect_gpu_count()

    config = vars(args)
    PipelineModule(config).run()
