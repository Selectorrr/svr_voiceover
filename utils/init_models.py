import os

import onnx_asr
from appdirs import user_cache_dir
from huggingface_hub import hf_hub_download

cache_dir = user_cache_dir("svr_tts", "SynthVoiceRu")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TQDM_POSITION"] = "-1"

REPO_ID = "selectorrrr/svr-tts-large"
MODEL_FILES = {
    "base": "svr_base_v2.onnx",
    "semantic": "svr_semantic.onnx",
    "encoder": "svr_encoder.onnx",
    "estimator": "svr_estimator.onnx",
    "vocoder": "svr_vocoder.onnx",
}
for key in MODEL_FILES.keys():
    hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES[key], cache_dir=cache_dir)

hf_hub_download(repo_id="selectorrrr/wav2vec2mos", filename="wav2vec2mos.onnx",
                cache_dir=user_cache_dir("svr_voiceover", "SynthVoiceRu"))

onnx_asr.load_model("onnx-community/whisper-large-v3-turbo")
