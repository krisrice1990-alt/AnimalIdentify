# Mammal-head on-device interface (contract for the app workstreams)

Model file: `mammal-head.tflite` (this directory; copied into the app at build time,
or downloaded once at runtime like the YAMNet weights — app-shell/YAMNet chats decide;
the file is tens of KB either way).

## Tensors

| | shape | dtype | meaning |
|---|---|---|---|
| input  | `[1, 521]` | float32 | mean-pooled YAMNet AudioSet posteriors over the clip |
| output | `[1, 6]`   | float32 | softmax scores, order below |

Class order: `0=wolf, 1=coyote, 2=moose, 3=bear, 4=dog, 5=background`

## Preprocessing (runs on-device, owned by the YAMNet module)

1. Take the audio engine's 3 s / 48 kHz mono clip (144,000 samples).
2. Resample to 16 kHz (anti-aliased; the YAMNet module already does this).
3. Slice into 15,600-sample windows, hop 7,680 (0.48 s) → 5 frames.
4. Run each frame through `yamnet.tflite` → 5 × 521 posteriors.
5. Mean-pool over frames → `(521,)` → reshape to `(1, 521)` → run `mammal-head.tflite`.

## Decision rule (thresholds are TUNED ON REAL DATA — do not ship pilot thresholds)

- Take argmax; report the class only if its score ≥ threshold (default 0.50,
  per-class thresholds to be set from the real-data evaluation report).
- Below threshold → "no confident detection" (never invent one).
- `dog` and `background` are explicit negative classes: a top-1 `dog` means
  "canine, not wolf/coyote" — surface it as such, never as wolf/coyote.

## Privacy

On-device only. Training data contains no human speech by construction
(dataset-chat rule); the app must still discard speech-like input and never
log or upload audio.
