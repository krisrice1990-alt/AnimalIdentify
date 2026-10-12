#!/usr/bin/env python3
"""Build the DRAFT dataset: clips/<species> -> draft-data/{train,val,test}/<class>.

Split is by SOURCE RECORDING (all 3 s chunks from one raw file land in the
same split) to prevent adjacent chunks leaking across splits. 70/15/15.
Labels are the species folder = UNVERIFIED draft labels (Kris hasn't
verified these yet). Do not ship anything trained on this.
"""
import json
import os
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.dirname(BASE)
CLIPS = os.path.join(DATASET, "clips")
OUT = os.path.join(BASE, "draft-data")
CLASSES = ["wolf", "fox"]
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 7


def main():
    import numpy as np
    rng = np.random.default_rng(SEED)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    counts = {}
    for cls in CLASSES:
        clipdir = os.path.join(CLIPS, cls)
        track = os.path.join(clipdir, "clips.jsonl")
        groups = []  # list of clip-name lists, one per source recording
        with open(track) as f:
            for line in f:
                rec = json.loads(line)
                if rec["clips"]:
                    groups.append(rec["clips"])
        if len(groups) < 3:
            raise SystemExit(f"ERROR: {cls} has only {len(groups)} source recordings; need at least 3 for train/val/test.")
        order = rng.permutation(len(groups)).tolist()
        n_val = max(1, int(round(len(groups) * VAL_FRAC)))
        n_test = max(1, int(round(len(groups) * TEST_FRAC)))
        if n_val + n_test >= len(groups):
            n_val = n_test = 1
        val_idx = set(order[:n_val])
        test_idx = set(order[n_val:n_val + n_test])
        for i, names in enumerate(groups):
            split = "val" if i in val_idx else ("test" if i in test_idx else "train")
            dest = os.path.join(OUT, split, cls)
            os.makedirs(dest, exist_ok=True)
            for nm in names:
                shutil.copy2(os.path.join(clipdir, nm), os.path.join(dest, nm))
                counts[(split, cls)] = counts.get((split, cls), 0) + 1
    for (split, cls), n in sorted(counts.items()):
        print(f"  {split}/{cls}: {n} clips")
    print("draft dataset ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
