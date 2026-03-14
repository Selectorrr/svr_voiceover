from pathlib import Path


class SpeakerProcessor:
    def __init__(self, audio_processor):
        self.audio = audio_processor

    def get_speaker_style(self, speaker, wave_24k):
        """
            Составляет сэмпл для опеределния тембра голоса и сохраняет его на диск
        """

        speaker_name = (speaker or "").strip()
        if not speaker_name:
            return self.audio.build_speaker_sample(None, wave_24k)

        voice = Path(f"workspace/voices/{speaker_name}.wav")

        style_wave_24k = self.audio.build_speaker_sample(voice, wave_24k)

        return style_wave_24k
