import numpy as np
import onnx_asr


class AsrModel:
    def __init__(self, providers, provider_options):
        self.session = onnx_asr.load_model("onnx-community/whisper-large-v3-turbo", providers=providers,
                                           provider_options=provider_options)

    def __call__(
            self,
            audio: np.ndarray,

    ):
        text = self.session.recognize(audio, language="ru")
        return text
