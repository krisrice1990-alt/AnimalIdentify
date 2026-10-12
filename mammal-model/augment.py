"""Waveform + feature-level augmentation for training.

Waveform augments (applied BEFORE YAMNet embedding extraction so the backbone
sees realistic variation):
  - time_shift: circular shift up to +/-250 ms
  - random_gain: +/-6 dB
  - add_noise: mix white/pink-ish noise at 10-25 dB SNR
  - polarity_flip

Feature-level augments (applied to the 521-dim YAMNet vector):
  - feature_noise: small Gaussian jitter (mimics embedding variation)
  - feature_dropout: randomly zero a fraction of dims (robustness)

No speech data is used anywhere; background-class clips are wind/rustle noise.
"""
import numpy as np

from config import SR_MODEL


def time_shift(x, rng, max_ms=250):
    s = rng.integers(-int(SR_MODEL * max_ms / 1000), int(SR_MODEL * max_ms / 1000) + 1)
    return np.roll(x, s)


def random_gain(x, rng, db=6.0):
    g = 10 ** (rng.uniform(-db, db) / 20.0)
    return np.clip(x * g, -1.0, 1.0)


def add_noise(x, rng, snr_db=(10, 25)):
    snr = rng.uniform(*snr_db)
    sig_pow = np.mean(x ** 2) + 1e-9
    noise = rng.standard_normal(len(x))
    noise = noise * np.sqrt(sig_pow / (np.mean(noise ** 2) * 10 ** (snr / 10)))
    return np.clip(x + noise, -1.0, 1.0)


def polarity_flip(x, rng, p=0.5):
    return -x if rng.random() < p else x


def augment_waveform(x, rng):
    """Random composition used during training."""
    x = time_shift(x, rng)
    x = polarity_flip(x, rng)
    x = random_gain(x, rng)
    if rng.random() < 0.7:
        x = add_noise(x, rng)
    return x.astype(np.float32)


def augment_feature(v, rng, noise_std=0.01, dropout_p=0.05):
    v = v + rng.normal(0, noise_std, v.shape).astype(np.float32)
    if dropout_p > 0:
        mask = (rng.random(v.shape) > dropout_p).astype(np.float32)
        v = v * mask / (1.0 - dropout_p)
    return v
