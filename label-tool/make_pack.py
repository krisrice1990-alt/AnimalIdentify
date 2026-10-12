#!/usr/bin/env python3
"""Build a labeling pack: N clips + label.html + manifest, zipped for the phone.

Usage:
    python3 make_pack.py --species wolf --count 15

Output:
    label-packs/<species>-batch<N>/  (clips/, clips.json, label.html)
    label-packs/<species>-batch<N>.zip
Packs are tracked in label-packs/packs.json so clips are never reused.
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.dirname(BASE)
PACKS_DIR = os.path.join(DATASET, "label-packs")
TRACK = os.path.join(PACKS_DIR, "packs.json")


def load_track():
    if os.path.exists(TRACK):
        with open(TRACK) as f:
            return json.load(f)
    return {"batches": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--count", type=int, default=15)
    args = ap.parse_args()

    raw = os.path.join(DATASET, "raw", args.species)
    files = sorted(
        f for f in os.listdir(raw)
        if os.path.isfile(os.path.join(raw, f)) and not f.startswith("metadata")
    )
    track = load_track()
    used = {c["src"] for b in track["batches"] for c in b["clips"]}
    fresh = [f for f in files if f not in used]
    if len(fresh) < args.count:
        print(f"only {len(fresh)} unused clips available, need {args.count}")
        sys.exit(1)
    chosen = fresh[: args.count]

    n = sum(1 for b in track["batches"] if b["species"] == args.species) + 1
    pack_name = f"{args.species}-batch{n}"
    pack_dir = os.path.join(PACKS_DIR, pack_name)
    clips_dir = os.path.join(pack_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    manifest_clips = []
    for i, src in enumerate(chosen, 1):
        _, dot, ext = src.rpartition(".")
        cid = f"{i:03d}"
        dst = f"{cid}.{ext}" if dot else cid
        shutil.copy2(os.path.join(raw, src), os.path.join(clips_dir, dst))
        manifest_clips.append({"id": cid, "file": dst, "src": src})

    with open(os.path.join(pack_dir, "clips.json"), "w") as f:
        json.dump(
            {"title": f"Label: {args.species} sounds (batch {n})", "clips": [
                {"id": c["id"], "file": c["file"]} for c in manifest_clips]},
            f,
        )
    shutil.copy2(os.path.join(BASE, "label.html"), pack_dir)

    zip_path = os.path.join(PACKS_DIR, pack_name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(pack_dir):
            for fn in fs:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, pack_dir))

    track["batches"].append(
        {"name": pack_name, "species": args.species, "clips": manifest_clips}
    )
    with open(TRACK, "w") as f:
        json.dump(track, f, indent=1)

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"pack: {zip_path} ({size_mb:.1f} MB, {len(chosen)} clips)")


if __name__ == "__main__":
    sys.exit(main())
