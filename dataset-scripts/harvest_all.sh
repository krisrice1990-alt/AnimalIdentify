#!/bin/bash
# Full database harvest. Resume-safe; run again any time.
cd "$(dirname "$0")"
python3 harvest_inat.py --species coyote --max 500
python3 harvest_inat.py --species moose --max 100
python3 harvest_inat.py --species bear --max 100
python3 harvest_inat.py --species dog --max 200
python3 harvest_inat.py --species wolf --max 300
python3 harvest_inat.py --species fox --max 300
echo "HARVEST ALL DONE"
for d in raw/*/; do echo "$d: $(find $d -type f ! -name 'metadata*' | wc -l) audio files"; done
