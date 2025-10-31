import multiprocessing
import os
from functools import cached_property

import GPUtil
import onnxruntime
from appdirs import user_cache_dir
from huggingface_hub import hf_hub_download
from svr_tts import SVR_TTS

from modules.AsrModel import AsrModel


class ModelFactory:
    """
        Возвращает закешированную в этом потоке модель.
    """

    def __init__(self, config):
        self.config = config

    @cached_property
    def svr_tts(self):
        import onnxruntime as ort
        result = SVR_TTS(self.config['api_key'], providers=self.config["providers"],
                      provider_options=self._get_provider_opts())
        print("providers:", ort.get_available_providers())  # должно быть 'CUDAExecutionProvider'
        sess = result.base_model
        print("active:", sess.get_providers())
        return result


    @cached_property
    def mos(self):
        os.environ["TQDM_POSITION"] = "-1"
        session = onnxruntime.InferenceSession(
            hf_hub_download(repo_id="selectorrrr/wav2vec2mos", filename="wav2vec2mos.onnx",
                            cache_dir=self._get_cache_dir()), providers=self.config["providers"],
            provider_options=self._get_provider_opts())
        return session

    @cached_property
    def asr(self):
        asr = AsrModel(providers=self.config["providers"], provider_options=self._get_provider_opts())
        return asr


    def _get_provider_opts(self):
        provider_options = []
        for provider in self.config["providers"]:
            if provider == "CUDAExecutionProvider":
                provider_options += [{"device_id": self._get_device_id()}]
            else:
                provider_options += [{}]

        return provider_options

    @staticmethod
    def get_job_n():
        current_process = multiprocessing.current_process()
        if current_process._identity:
            job_n = current_process._identity[0] - 1
        else:
            job_n = 0
        return job_n

    @staticmethod
    def _get_device_id():
        job_n = ModelFactory.get_job_n()
        gpu_n = job_n % (GPUtil.getGPUs().__len__() or 1)
        return gpu_n

    def _get_cache_dir(self) -> str:
        cache_dir = user_cache_dir("svr_voiceover", "SynthVoiceRu")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _detect_providers(self):
        providers = []
        try:
            # вернёт список GPU-объектов, даже если nvidia-smi не в PATH
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                # gpu.id — порядковый номер устройства
                providers.append(f"CUDAExecutionProvider:{gpu.id}")
        except Exception:
            # если что-то пойдёт не так — просто не добавим CUDA
            pass

        providers.append("CPUExecutionProvider")
        return providers
