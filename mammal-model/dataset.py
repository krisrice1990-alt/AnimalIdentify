"""Dataset manifest + feature caching.

Expected layout (dataset-chat contract):
    <root>/train/<class>/*.wav
    <root>/val/<class>/*.wav
    <root>/test/<class>/*.wav        (48 kHz WAVs)

Feature kinds:
    yamnet  - 521-dim mean-pooled YAMNet AudioSet posteriors (transfer path)
    logmel  - 64-dim pooled log-mel vector (loop-validation / fallback path)

Cache: <root>/cache_<kind>.npz  with X (n, d) float32, y (n,) int64,
       split (n,) int8 (0=train,1=val,2=test), ids (n,) str.
"""
import argparse
import os
import numpy as np

from config import CLASSES, CLASS_TO_IDX, SEED
from features import load_wav_mono_16k, pooled_logmel_vector

SPLIT_IDX = {"train": 0, "val": 1, "test": 2}


def scan(root):
    items = []
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".wav"):
                    items.append((os.path.join(d, f), CLASS_TO_IDX[cls], SPLIT_IDX[split]))
    return items


def build_cache(root, kind="yamnet", augment_train=False, seed=SEED):
    items = scan(root)
    if not items:
        raise SystemExit(f"no WAVs found under {root}")
    print(f"found {len(items)} clips under {root}")

    extractor = None
    if kind == "yamnet":
        from yamnet_extract import YamnetFeatureExtractor
        extractor = YamnetFeatureExtractor()

    Xs, ys, ss, ids = [], [], [], []
    rng = np.random.default_rng(seed)
    for i, (path, label, split) in enumerate(items):
        x = load_wav_mono_16k(path)
        reps = [(x, split)]
        if augment_train and split == 0:
            from augment import augment_waveform
            # 1 augmented twin per training clip keeps cache size manageable
            reps.append((augment_waveform(x, rng), split))
        for xx, ssp in reps:
            if kind == "yamnet":
                v = extractor.clip_vector(xx)
            elif kind == "logmel":
                v = pooled_logmel_vector(xx)
            else:
                raise ValueError(kind)
            Xs.append(v)
            ys.append(label)
            ss.append(ssp)
            ids.append(os.path.relpath(path, root))
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(items)}")
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    s = np.array(ss, dtype=np.int64)
    out = os.path.join(root, f"cache_{kind}.npz")
    np.savez_compressed(out, X=X, y=y, split=s,
                        ids=np.array(ids, dtype=object))
    for sp, name in [(0, "train"), (1, "val"), (2, "test")]:
        print(f"  {name}: {(s == sp).sum()} vectors, dim={X.shape[1]}")
    print("cache ->", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kind", choices=["yamnet", "logmel"], default="yamnet")
    ap.add_argument("--augment-train", action="store_true")
    args = ap.parse_args()
    build_cache(args.root, kind=args.kind, augment_train=args.augment_train)
