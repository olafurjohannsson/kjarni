import numpy as np
import librosa

audio, sr = librosa.load("crates/kjarni-models/examples/hideyowife.wav", sr=16000, duration=30.0)
audio = librosa.util.fix_length(audio, size=480000)

# Pad for center=True
padded = np.pad(audio, 200, mode='reflect')

# Frame 100
start = 100 * 160  # hop_length=160
frame = padded[start:start+400]
window = librosa.filters.get_window('hann', 400, fftbins=True)
windowed = frame * window

print("=== FFT Debug (frame 100) ===")
print(f"Windowed[:5]: {windowed[:5]}")

fft_result = np.fft.rfft(windowed)
magnitudes = np.abs(fft_result)
powers = magnitudes ** 2

print(f"Magnitudes[:10]: {magnitudes[:10]}")
print(f"Powers[:10]: {powers[:10]}")