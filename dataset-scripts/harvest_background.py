#!/usr/bin/env python3
"""Harvest background/ambient noise from iNaturalist for Animal Identify.

Sound observations filed under non-animal taxa (plants, fungi) are almost
always ambient: wind, rain, rustling leaves, distant traffic — exactly the
"no animal" material the classifier needs for its background class.

Reuses harvest_inat.harvest() so licensing (CC only), metadata, and
resume-safety are identical to the animal harvests.

Usage:
    python3 harvest_background.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_inat import harvest  # noqa: E402

# (taxon, max) — Plantae first, Fungi as backup
TARGETS = [("Plantae", 70), ("Fungi", 40)]

if __name__ == "__main__":
    total = 0
    for taxon, max_obs in TARGETS:
        total += harvest("background", taxon, max_obs)
    print(f"[background] TOTAL kept={total}", flush=True)
