import re

import numpy as np
import onnx_asr
from onnx_asr.asr import TimestampedResult

from modules.TextProcessor import similarity


class AsrModel:
    def __init__(self, providers, provider_options):
        self.session = onnx_asr.load_model("alphacep/vosk-model-ru", providers=providers, provider_options=provider_options).with_timestamps()

    def recognize(
            self,
            audio: np.ndarray,

    ):
        result = self.session.recognize(audio, language="ru")
        return result

