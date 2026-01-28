import multiprocessing
import os
from functools import cached_property

import GPUtil
import onnxruntime
import torch
from appdirs import user_cache_dir
from huggingface_hub import hf_hub_download
from svr_tts import SVR_TTS

from modules.AsrModel import AsrModel
from modules.VcCV3 import VcCV3


class ModelFactory:
    """
        Возвращает закешированную в этом потоке модель.
    """

    def __init__(self, config):
        self.config = config

    @cached_property
    def svr_tts(self) -> 'SVR_TTS':
        if self.config['vc_type'] == 'default':
            vc_model = VcCV3(self.config, torch.device(f"cuda:{self._get_device_id()}"))
            # vc_model = VcS3Gen(self.config, torch.device(f"cuda:{self._get_device_id()}"))
            vc_type = None
        else:
            vc_model = None
            vc_type = self.config['vc_type']

        result = SVR_TTS(self.config['api_key'], providers=self.config["providers"],
                      provider_options=self._get_provider_opts(), user_models_dir=self.config['user_models_dir'],reinit_every=self.config['reinit_every'], prosody_cond=self.config['prosody_cond'],
                         vc_func=vc_model, vc_type=vc_type, min_prosody_len=self.config['min_prosody_len'],                         speed_search_attempts=self.config['speed_search_attempts'],
                         speed_match_tolerance_pct=self.config['max_extra_speed'],
                         speed_clip_min=self.config['speed_clip_min'],
                         speed_clip_max=self.config['speed_clip_max'],
                         speed_adjust_step_pct=self.config['speed_adjust_step_pct'])
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

