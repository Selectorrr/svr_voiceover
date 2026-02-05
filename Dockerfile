# =========================
# runtime
# =========================
FROM selector/voiceover:env AS runtime

RUN python -m pip install -c constraints.txt \
      svr_tts==0.12.0

COPY . /workspace/SynthVoiceRu

ENTRYPOINT ["python", "entrypoint.py"]