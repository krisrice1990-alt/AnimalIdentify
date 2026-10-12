"""Shared configuration for the mammal-classifier training pipeline.

Classes follow the dataset-chat contract:
  wolf / coyote / moose / bear / dog (negative) / background
Dataset layout expected from the dataset chat:
  ~/workspace/field-listener/mammal-dataset/<split>/<class>/*.wav   (48 kHz WAV)
"""

CLASSES = ["wolf", "fox"]  # DRAFT-ONLY 2-class subset (unverified labels)
N_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

SR_NATIVE = 48000          # what the audio engine / dataset chat produce
SR_MODEL = 16000           # everything downstream runs at 16 kHz
CLIP_SEC = 3.0
CLIP_SAMPLES_16K = int(SR_MODEL * CLIP_SEC)   # 48000

# YAMNet TFLite framing (matches vendor/yamnet.tflite: 15600-sample input)
YAMNET_WIN = 15600         # ~0.975 s analysis window
YAMNET_HOP = 7680          # 0.48 s hop -> ~5 frames per 3 s clip
YAMNET_N_SCORES = 521      # AudioSet class posteriors; this TFLite has no embedding output

# Log-mel (fallback path / pilot loop-validation features)
N_MELS = 32
MEL_WIN_MS = 25.0
MEL_HOP_MS = 10.0

# MLP head hyper-parameters (numpy trainer)
HIDDEN_DIM = 128
LEARNING_RATE = 3e-3
BATCH_SIZE = 64
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 10
DROPOUT_P = 0.3

SEED = 7
