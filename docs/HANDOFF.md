# Mammal Listener — Handoff for ChatGPT

## What this is
A Home Assistant add-on that listens for **wolf, coyote, moose, bear**
(plus dog/background negatives) and reports detections to Home Assistant
over MQTT with auto-discovery. Standalone wildlife sound identification with Home Assistant integration.
All code + AI training database pipeline. Built 2026-10-12 by Muse for Kris.

## The add-on (`ha-mammal-addon/`)
- `mammal_listener/` — complete, installable HA add-on (config.yaml,
  build.yaml for aarch64/amd64/armv7, Dockerfile, s6 service, `listener/detector.py`).
- Pipeline: mic 48 kHz → 3 s windows → energy gate → resample 16 kHz →
  YAMNet (521 AudioSet posteriors, mean-pooled) → `mammal-head.tflite` →
  softmax over [wolf, coyote, moose, bear, dog, background].
- Publishes MQTT (TLS, port 8883 default) + HA auto-discovery sensors.
- Audio never leaves the machine; only species/confidence/timestamp published.
- `repository.yaml` included so the folder works as a custom HA add-on repo.
- **Status:** packaged and installable, BUT the trained `mammal-head.tflite`
  does not exist yet — the add-on refuses to start without it (says so in log).
  Training it is the main open job (see Database below).

## The AI database (`mammal-dataset/` + `mammal-model/`)
Goal: train `mammal-head.tflite` (tiny MLP: 521 → 128 → 6, tens of KB).

- `mammal-dataset/harvest_inat.py` — harvests real CC-licensed animal sounds
  from iNaturalist API (resume-safe, CC-only: cc0/cc-by/cc-by-nc/cc-by-sa/cc-by-nc-sa).
  Species: wolf, coyote, moose, bear, fox, dog (`Canis familiaris`).
  Run: `python3 harvest_inat.py --species coyote --max 500` or `--all`.
  iNat audio counts: wolf 786, coyote 3942, moose 34, bear 35, fox 2000, dog ~596.
  Moose/bear are thin — may not train reliably; wolf/coyote are the strong classes.
- `mammal-dataset/make_clips.py` — converts raw mp3/m4a → 3 s / 48 kHz mono
  WAV chunks into `clips/<species>/` (drops near-silent chunks, resume-safe).
- `mammal-dataset/label_tool/` — Kris is the human verifier: `make_pack.py`
  builds 15-clip labeling packs (tracked in `label-packs/packs.json`, never reused),
  `label.html` is the tap-to-label page. **wolf-batch1 (15 clips) was sent to
  Kris 2026-10-12 — labels still pending.** Nothing trains on unverified clips.
- `mammal-model/` — the training pipeline: `dataset.py` (builds YAMNet/log-mel
  feature caches from `<root>/{train,val,test}/<class>/*.wav`), `train.py`
  (numpy MLP), `evaluate.py`, `export_tflite.py`. Contract in `INTERFACE.md`.
- `mammal-dataset/draft/` — a 2-class (wolf-vs-fox) working draft of the full
  pipeline on already-harvested clips. Clearly marked DRAFT, unverified labels,
  do-not-ship. Proves the mechanics end to end.

## Hard rules (from Kris — do not break)
1. **No fake training data.** Real verified recordings only. `synth.py` is
   pipeline-validation only, never training data.
2. **CC licenses only** for harvested audio (ND variants excluded — training
   is adaptation).
3. **Kris verifies labels** — he volunteered ("what if I hear the animal sounds
   can I say what the sounds animal is"). Packs go to him; training waits on him.
4. **Hands off his Home Assistant** — never connect to or change his HA setup.
   This chat/pack is Q&A + build only.
5. **Credit-conscious** — work economically, no redundant rebuilds.
6. **Nothing ships/posts without his explicit go-ahead** (GitHub push and
   HA install both on hold until he says).

## What's running right now (2026-10-12 ~14:05 NDT)
- Harvest: coyote in progress (~190/500), then moose, bear, dog, wolf/fox top-up.
  Resume-safe — just re-run `harvest_all.sh`.
- Clips: wolf 635 + fox 908 done. More species to follow after harvest.
- Draft training (wolf-vs-fox): was running; may or may not have finished —
  check `mammal-dataset/draft/runs/draft/` and `draft_run.log`.

## Suggested next steps for ChatGPT
1. When harvest finishes: `make_clips.py --all`, then build label packs for
   coyote/moose/bear/dog (`make_pack.py --species X --count 15`) and hand them
   to Kris for verification.
2. Collect his labels (including pending wolf-batch1), assemble
   `train/val/test` per `dataset.py` layout, build the YAMNet cache
   (needs a TFLite runtime + `model-verify/yamnet.tflite`-equivalent;
   signature: input [15600] float32 → output [1,521] float32).
3. Train the 6-class head per `mammal-model/README.md`, evaluate, export
   `mammal-head.tflite` → that file goes to HA `/share/` and the add-on works.
4. Note: BirdNET v2.4 (the phone app's model) already ships wolf+coyote labels
   (`Canis lupus_Gray Wolf`, `Canis latrans_Coyote`) — a legit fallback detector
   for those two species with zero training if the head stalls on thin data.

## Repo layout in this zip
```
ha-mammal-addon/        the add-on (mammal_listener/ + repository.yaml + README.md)
mammal-dataset/         harvest + clips + labeling (NO raw audio; re-harvestable)
mammal-model/           training pipeline scripts + INTERFACE.md
sample-clips/           3 wolf + 3 fox 3 s WAVs (format reference)
HANDOFF.md              this file
```
