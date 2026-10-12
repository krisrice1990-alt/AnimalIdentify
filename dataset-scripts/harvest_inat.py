#!/usr/bin/env python3
"""Harvest training audio from the iNaturalist API into mammal-dataset/raw/.

One directory per species with the downloaded audio files plus a
metadata.jsonl describing every file (observation id, quality grade,
license, taxon). Only sounds with Creative Commons licenses are kept;
all-rights-reserved material is skipped. Resume-safe: existing files
listed in metadata.jsonl are not re-downloaded.

Usage:
    python3 harvest_inat.py --species wolf --max 200
    python3 harvest_inat.py --all --max 500
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")

SPECIES = {
    "wolf": "Canis lupus",
    "coyote": "Canis latrans",
    "moose": "Alces alces",
    "bear": "Ursus americanus",
    "fox": "Vulpes vulpes",
    "dog": "Canis familiaris",
}

# CC licenses we accept for training data. ND variants excluded because
# training a model is an adaptation of the work.
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}

API = "https://api.inaturalist.org/v1/observations"


def api_get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "FieldListener-Dataset/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "FieldListener-Dataset/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk)


def ext_for(file_url, content_type):
    path = urllib.parse.urlparse(file_url).path
    _, dot, ext = path.rpartition(".")
    if dot and len(ext) <= 5:
        return "." + ext.lower()
    return {"audio/mp4": ".m4a", "audio/mpeg": ".mp3", "audio/x-m4a": ".m4a"}.get(
        (content_type or "").split(";")[0].strip(), ".m4a"
    )


def harvest(species, taxon_name, max_obs):
    outdir = os.path.join(RAW, species)
    os.makedirs(outdir, exist_ok=True)
    meta_path = os.path.join(outdir, "metadata.jsonl")

    seen = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    pass

    kept = skipped_license = skipped_seen = 0
    page = 1
    total = None
    with open(meta_path, "a") as meta:
        while True:
            data = api_get(
                {
                    "taxon_name": taxon_name,
                    "sounds": "true",
                    "per_page": 200,
                    "page": page,
                    "order": "desc",
                    "order_by": "created_at",
                }
            )
            if total is None:
                total = data.get("total_results", 0)
                print(f"[{species}] {total} observations with sounds on iNaturalist", flush=True)
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                if kept >= max_obs:
                    break
                for snd in obs.get("sounds") or []:
                    lic = (snd.get("license_code") or obs.get("license_code") or "").lower()
                    if lic not in ALLOWED_LICENSES:
                        skipped_license += 1
                        continue
                    file_url = snd.get("file_url")
                    if not file_url:
                        continue
                    fname = f"{obs['id']}_{snd['id']}{ext_for(file_url, snd.get('file_content_type'))}"
                    if fname in seen:
                        skipped_seen += 1
                        continue
                    dest = os.path.join(outdir, fname)
                    try:
                        download(file_url, dest)
                    except Exception as e:  # noqa: BLE001 - keep harvesting on failures
                        print(f"[{species}] download failed {file_url}: {e}", flush=True)
                        continue
                    meta.write(
                        json.dumps(
                            {
                                "file": fname,
                                "obs_id": obs["id"],
                                "sound_id": snd["id"],
                                "quality_grade": obs.get("quality_grade"),
                                "license": lic,
                                "taxon": taxon_name,
                                "observed_on": obs.get("observed_on"),
                            }
                        )
                        + "\n"
                    )
                    meta.flush()
                    seen.add(fname)
                    kept += 1
                    if kept >= max_obs:
                        break
                    time.sleep(0.3)  # be polite to static hosting
            if kept >= max_obs:
                break
            if page * 200 >= (total or 0):
                break
            page += 1
            time.sleep(1.0)  # iNaturalist asks ~1 req/sec for sustained use
    print(
        f"[{species}] done: kept={kept} skipped_license={skipped_license} "
        f"skipped_seen={skipped_seen}",
        flush=True,
    )
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", choices=list(SPECIES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max", type=int, default=200, dest="max_obs")
    args = ap.parse_args()

    targets = list(SPECIES.items()) if args.all else [(args.species, SPECIES[args.species])]
    if not args.all and not args.species:
        ap.error("give --species NAME or --all")
    for species, taxon in targets:
        harvest(species, taxon, args.max_obs)


if __name__ == "__main__":
    sys.exit(main())
