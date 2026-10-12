# Animal Identify

Offline wildlife sound identification — detects wolves, coyotes, moose, bears, dogs
(and background) from microphone audio, with Home Assistant integration.

## How it works

```
mic 48 kHz mono → 3 s windows → energy gate (cheap) →
resample 16 kHz → YAMNet frames → mean-pooled posteriors →
mammal-head classifier (tiny TFLite) → MQTT → Home Assistant
```

Only windows that pass the energy gate reach YAMNet, so a quiet room costs
almost nothing. The classifier head is ~67 KiB quantized — the whole pipeline
runs comfortably on a Raspberry Pi.

## Repo layout

- `mammal-model/` — training pipeline (YAMNet features → small classifier head →
  TFLite export). This is the repaired canonical version (dropout-mask backprop
  fix, fail-safes on empty/NaN splits, recording-grouped train/val/test split).
- `dataset-scripts/` — `harvest_inat.py` (iNaturalist CC-licensed recordings,
  resume-safe), `make_clips.py` (raw → 3 s / 48 kHz mono WAVs, drops silence),
  `harvest_all.sh`.
- `label-tool/` — `label.html` (tap-to-label page for human verification) +
  `make_pack.py` (builds ~15-clip label packs). Clips are never reused across packs.
- `ha-addon/` — Home Assistant add-on (`mammal_listener`): s6 service, mic
  capture, energy gate → YAMNet → mammal-head → MQTT + HA discovery. Needs a
  trained `mammal-head.tflite` in `/share/`; exits cleanly if absent.
- `docs/` — handoff notes and status write-ups.

## Status (2026-10-12)

- Draft 2-class (wolf vs fox) pipeline PROVEN end-to-end: 635 wolf + 908 fox
  clips, recording-grouped 70/15/15 split, **test accuracy 77.7% / macro-F1 0.771**,
  exported `draft-head.tflite` (67 KiB quantized), verified through the real file.
- Real 6-class model (wolf, coyote, moose, bear, dog, background) awaits human
  labels — see `docs/STATUS-FOR-CHATGPT.md`.
- Dataset so far: coyote 1540 clips, fox 908, wolf 635 (+ dog/moose/bear converting).
  All real iNaturalist recordings, CC-licensed. No synthetic training data, ever.

## Rules

- Real verified recordings only. No synthetic/fake clips in training sets.
- Everything runs offline. No audio ever leaves the device.
