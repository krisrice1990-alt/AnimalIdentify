# Mammal Listener — Home Assistant App

On-device AI that listens for **wolf, coyote, moose and dog** sounds and
reports detections to Home Assistant over MQTT — like BirdNET-Go, but for
animals. No audio ever leaves the device — only detection metadata
(animal name + confidence + timestamp).

## What it does

- Captures microphone audio in 3-second windows (48 kHz mono)
- Cheap energy gate skips silence (a quiet porch costs almost no CPU)
- YAMNet feature extraction → trained 4-class classifier (66.6% accuracy
  on human-verified audio; wolf↔dog is the hardest pair)
- Publishes BirdNET-Go-style MQTT payloads: the same PascalCase field
  names (`CommonName`, `ScientificName`, `Confidence`, `Date`, `Time`)
  on retained topics, so anything that consumes BirdNET-Go detections
  understands these too
- HA MQTT auto-discovery: one sensor per animal + online/offline status

## Requirements

- Home Assistant OS or Supervised (needs the Supervisor + Apps panel)
- A microphone on the HA host (USB mic works)
- The **Mosquitto broker** app installed (for MQTT). Mammal Listener reads
  its credentials automatically from the Supervisor — no manual config.

## Install

1. In Home Assistant: **Settings → Apps → ⋮ → Repositories**
2. Add this repository URL:
   `https://github.com/krisrice1990-alt/AnimalIdentify`
3. Find **Mammal Listener** and click **Install**
4. Go to the **Configuration** tab. Defaults work if Mosquitto is
   installed (`mqtt_auto_config: true`). Otherwise fill in your broker
   host/user/password manually.
5. Click **Start**. Watch the **Log** tab — you should see:
   `MQTT auto-config: broker core-mosquitto:1883 ...` then `Listening...`
6. Four new sensors appear automatically:
   `sensor.mammal_listener_wolf`, `..._coyote`, `..._moose`, `..._dog`

## MQTT topics

Base topic (configurable): `mammal_listener`

| Topic | Retained | Contents |
|---|---|---|
| `mammal_listener/status` | yes | `online` / `offline` |
| `mammal_listener/detections/wolf` | yes | last wolf detection JSON |
| `mammal_listener/detections/coyote` | yes | last coyote detection JSON |
| `mammal_listener/detections/moose` | yes | last moose detection JSON |
| `mammal_listener/detections/dog` | yes | last dog detection JSON |
| `mammal_listener/detections/last` | yes | most recent detection, any animal |

Detection payload — same shape as BirdNET-Go (PascalCase), plus
snake_case equivalents for BirdNET-Pi-style consumers:

```json
{
  "CommonName": "Coyote", "ScientificName": "Canis latrans",
  "Confidence": 0.8234, "Date": "2026-10-12", "Time": "14:22:10",
  "common_name": "Coyote", "scientific_name": "Canis latrans",
  "confidence_score": 0.8234,
  "date": "2026-10-12", "time": "14:22:10",
  "Source": "animal-identify"
}
```

Unlike BirdNET-Go there are no `BirdImage`/`ClipName` fields — by design,
no audio is recorded to disk or sent anywhere.

## Configuration options

| Option | Default | What it does |
|---|---|---|
| `mqtt_auto_config` | `true` | Read broker credentials from the Supervisor (needs Mosquitto app) |
| `mqtt_host` / `mqtt_port` / `mqtt_tls` / `mqtt_user` / `mqtt_password` | — | Manual broker settings (used if auto-config finds nothing) |
| `base_topic` | `mammal_listener` | MQTT topic prefix |
| `model_path` | `/app/models/mammal-head.tflite` | Classifier model; point at `/share/mammal-head.tflite` to use your own trained model |
| `energy_threshold_db` | `-45` | Windows quieter than this are skipped |
| `confidence_threshold` | `0.6` | Detections below this are ignored |
| `cooldown_minutes` | `10` | Minimum gap between repeat detections of the same animal |

## What's tested vs not

**Tested (2026-10-12):**
- Model file loads and has the exact I/O shape the detector expects
  (`[1, 521]` → `[1, 4]`)
- Detector Python compiles clean; all pure-logic tests pass (energy
  gate math, Supervisor MQTT auto-config parsing, dual-dialect payload
  construction, HA discovery config, options loading)
- `config.yaml` parses and options/schema keys match
- Dockerfile COPY sources all exist; YAMNet download URL verified live
  (redirects to Google Cloud Storage, returns valid TFLite bytes)
- MQTT payload fields verified field-by-field against BirdNET-Go's
  published format (`CommonName`, `ScientificName`, `Confidence`,
  `Date`, `Time`, `Source`); connection pattern (retained messages,
  Supervisor broker lookup) matches the BirdNET-Go HA app approach

**NOT tested — do not assume these work yet:**
- Docker image build (no Docker in this environment)
- Install on a real Home Assistant host
- Microphone capture on HA hardware
- End-to-end: real animal sound → MQTT → HA sensor
- The Supervisor MQTT auto-config against a live Mosquitto app

Install it, check the log, and if `Listening...` appears with no errors,
make a loud howl near the mic and watch for a detection.

## Privacy

All inference runs on the HA host. Only small JSON detection messages go
over MQTT. No audio is recorded to disk or sent anywhere.

## Files

- `config.yaml` — app metadata, options, schema
- `Dockerfile` — builds the image (bundles YAMNet + the 4-class model)
- `build.yaml` — base images per architecture
- `listener/detector.py` — the whole pipeline
- `listener/requirements.txt` — Python deps
- `models/mammal-head.tflite` — trained 4-class classifier (bundled)
- `rootfs/` — s6 service definition
