# Mammal classifier training (Field Listener)

Custom on-device classifier for **wolf / coyote / moose / bear**, with
**dog** and **background** as explicit negative classes. Transfer learning:
a small MLP head on top of YAMNet's 521 AudioSet posteriors
(mean-pooled over the clip). Fallback path (small CNN on log-mel) is
documented in `EVALUATION_REPORT.md` and activates only if transfer learning
underperforms on real data.

## Layout

- `config.py` — classes, sample rates, framing, hyper-parameters
- `features.py` — WAV loading, 48 kHz→16 kHz resample, YAMNet framing, log-mel
- `yamnet_extract.py` — runs `vendor/yamnet.tflite` (same SHA-256 the
  mammal-sounds module verifies at runtime) via ai_edge_litert
- `synth.py` — synthetic pilot-data generator (caricatured signatures;
  pipeline validation only, NOT real animal audio)
- `augment.py` — waveform + feature augmentation (time-shift, gain, noise mix,
  polarity flip, feature jitter/dropout)
- `dataset.py` — scans `<root>/<split>/<class>/*.wav`, caches features to npz
- `train.py` — numpy MLP head (Adam, class-weighted CE, early stopping on
  val macro-F1)
- `evaluate.py` — per-class precision/recall/F1 + confusion matrices
- `export_tflite.py` — Keras rebuild → dynamic-range-quantized TFLite + parity check
- `INTERFACE.md` — the on-device contract for the app workstreams
- `vendor/yamnet.tflite` — official TF Hub artifact (4.1 MB incl. labels zip)

## Quickstart (pilot)

```bash
./run_pilot.sh
```

This generates synthetic clips, builds two feature caches (log-mel for
loop-validation, YAMNet for transfer-path mechanics), trains both heads,
evaluates, and exports the TFLite head.

## Real data (when the dataset chat delivers)

```bash
python3 dataset.py --root ./dataset \
    --kind yamnet --augment-train
python3 train.py --cache ./dataset/cache_yamnet.npz \
    --out-dir runs/real_yamnet
python3 evaluate.py --weights runs/real_yamnet/weights.npz \
    --cache ./dataset/cache_yamnet.npz \
    --out-dir runs/real_yamnet --title "real-data yamnet head"
python3 export_tflite.py --weights runs/real_yamnet/weights.npz \
    --cache ./dataset/cache_yamnet.npz \
    --out runs/real_yamnet/mammal-head.tflite
```

Expected dataset layout: `<root>/train|val|test/<class>/*.wav` at 48 kHz.
No human speech anywhere in training data (standing rule).

## Honesty notes

- Pilot accuracy numbers validate the *pipeline*, not real-world performance.
- Wolf vs coyote vs dog is genuinely hard; the report must show the confusion
  matrix, not hide it. If a class can't be learned reliably, the app uses the
  YAMNet fallback for that animal instead of shipping a guesser.
