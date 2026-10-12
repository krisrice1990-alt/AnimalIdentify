# Mammal Listener — status write-up for ChatGPT (2026-10-12 ~14:35 NDT)

## Your repaired version: received, verified, adopted
- Unpacked `MammalListener-repaired-handoff_4_m1xh.zip`, read `FIXES_AND_VALIDATION.md`.
- Verified the dropout bug against the original source before trusting it: confirmed real —
  `forward()` cached the masked `a1` but the backward pass applied only the ReLU derivative,
  never the dropout mask. Your fix is correct.
- Diffed your `train.py` against the local copy: functionally identical dropout fix
  (naming only), plus your fail-safes (empty/NaN splits, <2 classes) and the TEST-NOT-RUN
  warning. Adopted your `train.py`, `export_tflite.py`, `make_draft_data.py` as canonical.
  My draft-only patches (2-class config, YAMNet model path) were untouched by your pass
  and remain in place.

## The draft is now finished (on your repaired code)
Full pipeline re-ran end to end: clips → YAMNet features → train → evaluate → TFLite export.

- Data: 635 wolf + 908 fox clips (3 s / 48 kHz mono), split 70/15/15 **by source recording**
  (your grouped split): train 1039, val 173, test 233 clips.
- Results (repaired trainer): **val acc 67.1% / macro-F1 0.670**, **test acc 77.7% / macro-F1 0.771**.
- Export: `draft-head.tflite`, **67 KiB quantized**, plus end-to-end verification running real
  audio → YAMNet → the actual `.tflite` file on the held-out test set: **77.3%** —
  matches the numpy eval within quantization noise. Parity holds.
- Artifacts: `mammal-dataset/draft/runs/draft/` (weights.npz, metrics.json,
  confusion_matrix_{val,test}.png, draft-head.tflite).

## Two more bugs found while finishing (both in export path, both fixed locally)
1. **Keras 3 layer indexing**: `export_tflite.py` used `model.layers[1]`/`[2]`, but Keras 3
   (TF 2.16) excludes the explicit `Input` layer from `model.layers` — indices were off by
   one and `fc2` received `W1`'s shape. Fixed with `model.get_layer("fc1"/"fc2")`.
   (Also applied to `mammal-model/export_tflite.py`? — NO, only the draft copy was fixed;
   the 6-class original still has the positional indexing. Fix it the same way before use.)
2. **TF 2.16 TFLite converter MLIR crash** (`LLVM ERROR: Failed to infer result type(s)`,
   `missing attribute 'value'` on `ReadVariableOp`): the converter chokes on Keras-3
   variables in this build, even unquantized. Worked around by exporting from a plain
   `tf.function` with weights as `tf.constant`s + `Optimize.DEFAULT` — converts cleanly.
   The committed `export_tflite.py` still uses the Keras path; it will need this workaround
   (or a pinned TF version) to actually export.

## Environment that works
Fresh venv (no system site-packages): `numpy==1.26.4`, `scipy==1.11.4`,
`ai_edge_litert`, `tensorflow-cpu==2.16.2`, `matplotlib`.
(`~/workspace/yamnet-venv2/`.) YAMNet artifact: `~/workspace/model-verify/yamnet.tflite`,
signature verified: input [15600] float32 → output [1,521] float32.

## Honest limits (unchanged)
- Labels are **unverified** — Kris has wolf-batch1 (15 clips) and hasn't returned labels.
  This draft proves the pipeline, not a shippable detector. Do not present 77% as a product claim.
- Draft is 2-class (wolf/fox); the real head is 6-class (wolf/coyote/moose/bear/dog/background).
- Moose/bear have ~35 iNat recordings each — thin; may not train reliably.

## Still open
- Harvest still running (coyote ~190/500, then moose/bear/dog): resume-safe via `harvest_all.sh`.
- Next: label packs for coyote/moose/bear/dog → Kris verifies → 6-class train on verified data.
- Kris's rules stand: real verified recordings only, CC licenses only, hands off his Home Assistant,
  nothing ships or posts without his explicit go-ahead.
