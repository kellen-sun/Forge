import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from numba import njit, prange

y, sr = librosa.load("noisy_audio.wav", sr=None)

S = librosa.stft(y)
mag, phase = np.abs(S), np.angle(S)

noise_est = np.percentile(mag, 90, axis=1)


@njit(parallel=True)
def spectral_gate(mag_spec, noise_floor, threshold=1.5):
    output = np.empty_like(mag_spec)
    for i in prange(mag_spec.shape[0]):  # frequency bins
        cutoff = noise_floor[i] * threshold
        for j in range(mag_spec.shape[1]):  # time frames
            output[i, j] = mag_spec[i, j] if mag_spec[i, j] > cutoff else 0
    return output


t = time.time()
denoised_mag = spectral_gate(mag, noise_est)
print(f"Time taken for denoising: {time.time() - t:.4f} seconds")
S_denoised = denoised_mag * np.exp(1j * phase)
y_denoised = librosa.istft(S_denoised)

sf.write("denoised_output.wav", y_denoised, sr)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(
    librosa.amplitude_to_db(mag, ref=np.max), y_axis="log", x_axis="time"
)
plt.title("Original")

plt.subplot(1, 2, 2)
librosa.display.specshow(
    librosa.amplitude_to_db(denoised_mag, ref=np.max), y_axis="log", x_axis="time"
)
plt.title("Denoised")
plt.tight_layout()
plt.show()
