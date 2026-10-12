#!/usr/bin/env python3
"""Convert harvested raw audio into 3 s / 48 kHz mono WAV clips.

    python3 make_clips.py --species wolf
    python3 make_clips.py --all

Reads raw/<species>/*.{mp3,m4a,wav,...}, splits each file into non-overlapping
3 s chunks, converts to 48 kHz mono 16-bit WAV, writes to clips/<species>/.

Resume-safe: skips clips whose source file already produced output (tracked
in clips/<species>/clips.jsonl). Near-silent chunks (peak < -50 dBFS) are
dropped — they teach the model nothing.
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
CLIPS = os.path.join(BASE, "clips")
CHUNK_S = 3
SR = 48000
SILENCE_DB = -50


def file_dbfs_peak(path):
    """Peak dBFS of a file via ffmpeg astats (cheap, no decode to disk)."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-af", "astats=metadata=1",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    peak = None
    for line in (p.stderr or "").splitlines():
        if "Peak level dB" in line:
            try:
                v = float(line.split(":")[-1].strip())
                peak = v if peak is None else max(peak, v)
            except ValueError:
                pass
    return peak


def split_species(species):
    rawdir = os.path.join(RAW, species)
    outdir = os.path.join(CLIPS, species)
    os.makedirs(outdir, exist_ok=True)
    track_path = os.path.join(outdir, "clips.jsonl")
    done = set()
    if os.path.exists(track_path):
        with open(track_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["src"])
                except (json.JSONDecodeError, KeyError):
                    pass

    files = sorted(
        f for f in os.listdir(rawdir)
        if os.path.isfile(os.path.join(rawdir, f)) and not f.startswith("metadata")
    )
    made = skipped = 0
    with open(track_path, "a") as track:
        for fn in files:
            if fn in done:
                skipped += 1
                continue
            src = os.path.join(rawdir, fn)
            peak = file_dbfs_peak(src)
            if peak is not None and peak < SILENCE_DB:
                track.write(json.dumps({"src": fn, "clips": [], "note": "silent"}) + "\n")
                continue
            # Split into 3 s mono 48 kHz chunks.
            tmp = os.path.join(outdir, "_chunk_%03d.wav")
            p = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                 "-ac", "1", "-ar", str(SR), "-f", "segment",
                 "-segment_time", str(CHUNK_S), tmp],
                capture_output=True, text=True,
            )
            if p.returncode != 0:
                print(f"  !! ffmpeg failed on {fn}: {p.stderr.strip()[:120]}")
                continue
            stem = os.path.splitext(fn)[0]
            names = []
            for cf in sorted(os.listdir(outdir)):
                if cf.startswith("_chunk_"):
                    # Keep only full 3 s chunks (drop short tail).
                    full = os.path.join(outdir, cf)
                    dur = subprocess.run(
                        ["ffprobe", "-hide_banner", "-loglevel", "error",
                         "-show_entries", "format=duration", "-of",
                         "default=noprint_wrappers=1:nokey=1", full],
                        capture_output=True, text=True,
                    )
                    try:
                        d = float((dur.stdout or "0").strip())
                    except ValueError:
                        d = 0
                    if d < CHUNK_S - 0.05:
                        os.remove(full)
                        continue
                    new = f"{stem}_{cf[7:10]}.wav"
                    os.rename(full, os.path.join(outdir, new))
                    names.append(new)
                    made += 1
            track.write(json.dumps({"src": fn, "clips": names}) + "\n")
    print(f"[{species}] clips made={made} files_skipped={skipped}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    species = [args.species] if args.species else sorted(
        d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d))
    )
    if args.all and args.species:
        ap.error("use --species or --all, not both")
    for sp in species:
        split_species(sp)


if __name__ == "__main__":
    sys.exit(main())
