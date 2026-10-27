"""Times one engine transcribing English speech with whisper-small, greedy, on
the CPU, in a fresh process. Prints JSON.

    python whisper_worker.py ENGINE AUDIO
ENGINE: kjarni | openai-whisper | faster-whisper | faster-whisper-int8
"""
import json, os, sys, time, resource

ENGINE, AUDIO = sys.argv[1], sys.argv[2]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = 3

t0 = time.perf_counter()
if ENGINE == "kjarni":
    sys.path.insert(0, os.path.join(REPO, "crates/kjarni-ffi/bindings/python"))
    import kjarni
    m = kjarni.Transcriber(model="whisper-small", language="en", quiet=True)
    run = lambda: m.transcribe_file(AUDIO).text
elif ENGINE == "openai-whisper":
    import whisper, torch
    m = whisper.load_model("small", device="cpu")
    run = lambda: m.transcribe(AUDIO, language="en", temperature=0.0, fp16=False, beam_size=None)["text"]
else:
    import numpy as np, wave
    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8" if ENGINE.endswith("int8") else "float32")
    # Decoded here and passed as samples: faster-whisper's own decoder needs a
    # PyAV older than the one installed. The file is 16 kHz mono PCM already,
    # so decoding is not part of what is being compared.
    with wave.open(AUDIO) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    run = lambda: " ".join(s.text for s in m.transcribe(samples, language="en", beam_size=1, temperature=0.0)[0])
load_ms = (time.perf_counter() - t0) * 1000

text = run()  # warm-up, and the transcript to score
times = []
for _ in range(RUNS):
    s = time.perf_counter(); run(); times.append(time.perf_counter() - s)

import wave
with wave.open(AUDIO) as w:
    duration = w.getnframes() / w.getframerate()
print(json.dumps(dict(engine=ENGINE, audio=os.path.basename(AUDIO), audio_secs=duration, load_ms=load_ms,
                      secs=min(times), realtime_x=duration / min(times), text=text.strip(),
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
