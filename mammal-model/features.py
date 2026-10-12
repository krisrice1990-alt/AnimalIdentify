"""Audio feature extraction. numpy/scipy only (no librosa in this sandbox).

- WAV loading (any sample rate) -> mono float32 @ 16 kHz
- YAMNet framing: 15600-sample windows, 7680-sample hop
- Log-mel spectrograms (fallback path + pilot loop-validation)
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import SR_MODEL, YAMNET_WIN, YAMNET_HOP, N_MELS, MEL_WIN_MS, MEL_HOP_MS


def load_wav_mono_16k(path):
    """Read a WAV file, mix to mono, resample to 16 kHz float32 in [-1, 1]."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if sr != SR_MODEL:
        from math import gcd
        if sr == 48000:
            data = resample_poly(data, 1, 3)
        else:
            g = gcd(sr, SR_MODEL)
            data = resample_poly(data, SR_MODEL // g, sr // g)
    peak = np.abs(data).max()
    if peak > 1.0:
        data = data / peak
    return data.astype(np.float32)


def frame_for_yamnet(x):
    """Split a 16 kHz clip into YAMNet's 15600-sample windows (hop 7680).

    Returns (n_frames, 15600) float32. Short clips are zero-padded.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < YAMNET_WIN:
        x = np.pad(x, (0, YAMNET_WIN - len(x)))
    starts = range(0, len(x) - YAMNET_WIN + 1, YAMNET_HOP)
    return np.stack([x[s:s + YAMNET_WIN] for s in starts]).astype(np.float32)


# ---------------- log-mel (numpy STFT + mel filterbank) ----------------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bins[m - 1], bins[m], bins[m + 1]
        if f1 > f0:
            fb[m - 1, f0:f1] = (np.arange(f0, f1) - f0) / max(1, f1 - f0)
        if f2 > f1:
            fb[m - 1, f1:f2] = (f2 - np.arange(f1, f2)) / max(1, f2 - f1)
    return fb


def logmel(x, sr=SR_MODEL, n_mels=N_MELS, win_ms=MEL_WIN_MS, hop_ms=MEL_HOP_MS):
    """Log-mel spectrogram, shape (n_frames, n_mels)."""
    x = np.asarray(x, dtype=np.float32)
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n_fft = win
    window = np.hanning(win)
    n_frames = 1 + (len(x) - win) // hop
    if n_frames < 1:
        x = np.pad(x, (0, win - len(x)))
        n_frames = 1
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * window
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2          # (frames, n_fft//2+1)
    mel = mel_filterbank(sr, n_fft, n_mels)                  # (n_mels, n_fft//2+1)
    mel_spec = spec @ mel.T
    return np.log(mel_spec + 1e-6).astype(np.float32)


def pooled_logmel_vector(x):
    """Compact clip descriptor for pilot loop-validation: mean+std over time.

    Returns (2*n_mels,) float32, z-normalized per clip.
    """
    m = logmel(x)                       # (frames, n_mels)
    v = np.concatenate([m.mean(axis=0), m.std(axis=0)])
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v.astype(np.float32)
