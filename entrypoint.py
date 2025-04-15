import argparse

from modules.PipelineModule import PipelineModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--ext', type=str, default="ogg", help='Формат результирующего ауидо')
    parser.add_argument('--tone_sample_len', type=int, default=7500, help='Длина сэмпла голоса')
    parser.add_argument('--api_key', type=str, help='Ваш ключ доступа к api', required=True)
    parser.add_argument('--batch_size', type=int, default=20, help='Размер батча')

    args = parser.parse_args()

    config = vars(args)
    PipelineModule(config).run()
