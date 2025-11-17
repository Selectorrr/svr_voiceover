import sys
from pathlib import Path

import soundfile

from entrypoint import get_def_providers
from modules.AudioProcessor import AudioProcessor
from modules.CsvProcessor import CsvProcessor
from modules.ModelFactory import ModelFactory
from modules.PipelineModule import SynthesisInput
from modules.TextProcessor import TextProcessor

config = {
    'api_key': sys.argv[2].split('=')[1],
    'csv_delimiter': ',',
    'ext': 'ogg',
    'path_filter': '',
    'tone_sample_len': 7500,
    'is_respect_mos': False,
    'is_strict_len': True,
    'is_use_voice_len': True,
    'providers': get_def_providers(),
    'user_models_dir': ''
}
text_module = TextProcessor()
csv_module = CsvProcessor(config, text_module)
audio_module = AudioProcessor(config)

factory = None
def _worker_init(config):
    global factory
    factory = ModelFactory(config)

_worker_init(config)

from collections import Counter, defaultdict


def group_top_speakers(items, top_n=30):
    if not items:
        return []

    total = len(items)
    counts = Counter(d["speaker"] for d in items)

    top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:top_n]
    top_set = {sp for sp, _ in top}

    groups = defaultdict(list)
    for d in items:
        sp = d["speaker"]
        if sp in top_set:
            groups[sp].append(d)

    result = []
    for sp, cnt in top:
        grp = groups[sp]

        # сортируем по 'text' от большего к меньшему
        grp_sorted = sorted(grp, key=lambda x: len(x["text"]), reverse=True)

        result.append({
            "speaker": sp,
            "count": cnt,
            "percent": round(cnt * 100.0 / total, 2),
            "items": grp_sorted,
        })

    return result

def build_voice(records, voice_count=10):
    voice_num = 1
    voices = {}
    for record in records:
        path = Path(f"/workspace/SynthVoiceRu/workspace/resources/{record['audio']}")
        # Прочитаем оригинальный аудио файл и его громкость для определения просодии
        try:
            raw_wave_norm, meta, raw_wave, raw_sr = audio_module.load_audio(str(path))
        except ValueError as e:
            print(e)
            return
        # Определяем текст для озвучки
        # text, is_accented = text_module.get_text(record)
        speaker = record["speaker"]

        voice = Path(f"/workspace/SynthVoiceRu/workspace/voices/{speaker}/{voice_num}/{speaker}.wav")
        style_wave = audio_module.build_speaker_sample(voice, raw_wave_norm, raw_sr)
        voices[voice_num] = style_wave, 24_000
        if (len(style_wave) / 24_000) * 1000 >= config['tone_sample_len']:
            voice_num += 1
        if voice_num > voice_count:
            break
    return voices


def infer(input):
    job_n = ModelFactory.get_job_n()
    waves = factory.svr_tts.synthesize(
        [input],
        tqdm_kwargs={
            'leave': False,
            'smoothing': 0,
            'position': job_n + 1,
            'postfix': f"job_n {job_n}",
            'dynamic_ncols': True
        }
    )
    return waves[0]


def main():
    _, all_records = csv_module.find_changed_text_rows_csv()
    groups = group_top_speakers(all_records)
    from tqdm import tqdm
    num = 0
    for group in tqdm(groups, smoothing=0):
        num += 1
        print(f"{num}/{len(groups)}")
        speaker = group['speaker']
        print(f"{speaker}: {len(group['items'])} элементов ({len(group['items']) / len(all_records) * 100:.1f}%)")
        voices = build_voice(group['items'], voice_count=10)
        for voice_num, style in voices.items():
            style_wave, style_sr = style
            record = group['items'][0]
            text, is_accented = text_module.get_text(record)

            path = Path(f"/workspace/SynthVoiceRu/workspace/resources/{record['audio']}")
            raw_wave_norm, meta, raw_wave, raw_sr = audio_module.load_audio(str(path))

            input = SynthesisInput(text=text, stress=is_accented,
                           timbre_wave_24k=style_wave,
                           prosody_wave_24k=audio_module.prepare_prosody(raw_wave_norm, raw_sr))

            sample_path = Path(f"/workspace/SynthVoiceRu/workspace/voices/{speaker}/{voice_num}/sample.wav")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            wave = infer(input)
            soundfile.write(str(sample_path), wave, 22_050)


    pass


if __name__ == '__main__':
    main()
