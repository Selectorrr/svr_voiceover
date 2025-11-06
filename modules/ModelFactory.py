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
                      provider_options=self._get_provider_opts(),
                      session_options=self._get_session_opts())
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
            provider_options=self._get_provider_opts(), sess_options=self._get_session_opts())
        return session

    @cached_property
    def asr(self):
        asr = AsrModel(providers=self.config["providers"], provider_options=self._get_provider_opts())
        return asr


    def _get_provider_opts(self):
        provider_options = []
        for provider in self.config["providers"]:
            if provider == "CUDAExecutionProvider":
                provider_options += [{
                    'device_id': self._get_device_id(),
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'sdpa_kernel': '2',
                    'use_tf32': '1',
                    # 'fuse_conv_bias': '1',
                    'cudnn_conv_use_max_workspace': '1',
                    'cudnn_conv1d_pad_to_nc1d': '1',
                    'tunable_op_enable': '0',
                    'tunable_op_tuning_enable': '0',
                    'tunable_op_max_tuning_duration_ms': 10,
                    # 'do_copy_in_default_stream': '0',
                    'enable_cuda_graph': '0',
                    'prefer_nhwc': '0',
                    'enable_skip_layer_norm_strict_mode': '0',
                    'use_ep_level_unified_stream': '0',
                }]

            else:
                provider_options += [{}]

        return provider_options

    def _get_session_opts(self):
        session_opts = onnxruntime.SessionOptions()
        session_opts.log_severity_level = 4  # fatal level = 4, it an adjustable value.
        session_opts.log_verbosity_level = 4  # fatal level = 4, it an adjustable value.
        session_opts.inter_op_num_threads = 8  # Run different nodes with num_threads. Set 0 for auto.
        session_opts.intra_op_num_threads = 8  # Under the node, execute the operators with num_threads. Set 0 for auto.
        session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
        session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
        session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
        session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
        session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
        session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
        session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        if "CPUExecutionProvider" in self.config["providers"]:
            session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        elif 'CUDAExecutionProvider' in self.config["providers"]:
            session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return session_opts

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
