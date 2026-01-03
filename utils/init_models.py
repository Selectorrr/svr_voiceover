import os

import onnx_asr
from appdirs import user_cache_dir
from huggingface_hub import hf_hub_download, snapshot_download

cache_dir = user_cache_dir("svr_tts", "SynthVoiceRu")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TQDM_POSITION"] = "-1"

REPO_ID = "selectorrrr/svr-tts-large"
MODEL_FILES = {
    "base": "svr_base_v5.onnx",
    "semantic": "svr_semantic.onnx",
    "encoder": "svr_encoder_v1.onnx",
    "style": "svr_style.onnx",
    "estimator": "svr_estimator.onnx",
    "cfe": "svr_cfe.onnx",
}

for key in MODEL_FILES.keys():
    hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES[key], cache_dir=cache_dir)

hf_hub_download(repo_id="selectorrrr/wav2vec2mos", filename="wav2vec2mos.onnx",
                cache_dir=user_cache_dir("svr_voiceover", "SynthVoiceRu"))

providers = ['CPUExecutionProvider']
onnx_asr.load_model("alphacep/vosk-model-ru", providers=providers).with_timestamps()

hf_hub_download(repo_id="BSC-LT/vocos-mel-22khz", filename="mel_spec_22khz_univ.onnx",
                cache_dir=user_cache_dir("svr_tts", "SynthVoiceRu"))

snapshot_download(
            repo_id="selectorrrr/svr-tts-large",
            repo_type="model",
            allow_patterns=["s3gen/**"],
            token=os.getenv("HF_TOKEN"),  # нужен только если репо приватное
        )