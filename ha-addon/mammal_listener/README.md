# Mammal Listener — Home Assistant App

On-device AI that listens for **wolf, coyote, moose and dog** sounds and
reports detections to Home Assistant over MQTT — like BirdNET-Go, but for
animals. No audio ever leaves the device — only detection metadata
(animal name + confidence + timestamp).

## What it does

- Captures audio in 3-second windows (48 kHz mono) from a microphone,
  an RTSP camera, or Frigate cameras
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
- An audio source: a microphone on the HA host (USB mic works), **or**
  an IP/security camera with a microphone, **or** the Frigate app
- The **Mosquitto broker** app installed (for MQTT). Mammal Listener reads
  its credentials automatically from the Supervisor — no manual config.

## Audio sources

Three options, picked with the `audio_source` / Frigate settings:

**1. Microphone (default).** Plug a USB mic into the HA host. Nothing to
configure — `audio_source: mic`.

**2. RTSP camera.** Any IP/security camera with a microphone that exposes
an RTSP stream. Set:
```yaml
audio_source: rtsp
rtsp_url: rtsp://192.168.1.100:554/stream1   # your camera's audio stream
```
The app pulls **audio-only** from the camera first (no video decoder slot
wasted on the camera — same approach as BirdNET-Go) and automatically
falls back to the full stream if the camera can't do audio-only.

**3. Frigate.** If you run the Frigate app, Mammal Listener can listen
through its cameras via Frigate's built-in go2rtc restream — no need to
dig up RTSP URLs. Set:
```yaml
frigate_enabled: true
frigate_host: ccab4aaf-frigate   # or its IP, e.g. 192.168.1.100
frigate_camera: driveway         # one camera...
frigate_cameras: [driveway, porch]  # ...or several at once
```
Each camera runs detection in parallel and every detection payload
names its camera (`Camera` / `camera` fields), so you can tell which
camera heard the animal.

Tip: `frigate_host` accepts the Frigate app's internal hostname
(`ccab4aaf-frigate`) or a plain IP address.

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
  "Source": "animal-identify",
  "Camera": "frigate:driveway", "camera": "frigate:driveway"
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
| `audio_source` | `mic` | `mic` or `rtsp` — where audio comes from |
| `rtsp_url` | `""` | Camera stream URL (when `audio_source: rtsp`) |
| `frigate_enabled` | `false` | Listen through Frigate cameras instead |
| `frigate_host` | `""` | Frigate hostname (e.g. `ccab4aaf-frigate`) or IP |
| `frigate_camera` | `""` | Single Frigate camera name |
| `frigate_cameras` | `[]` | Multiple Frigate camera names (each runs detection) |

## What's tested vs not

**Tested (2026-10-12):**
- Model file loads and has the exact I/O shape the detector expects
  (`[1, 521]` → `[1, 4]`)
- Detector Python compiles clean; all pure-logic tests pass (energy
  gate math, Supervisor MQTT auto-config parsing, dual-dialect payload
  construction, HA discovery config, options loading)
- `config.yaml` parses and options/schema keys match (incl. new
  `audio_source` / `rtsp_url` / `frigate_*` options)
- Dockerfile COPY sources all exist; YAMNet download URL verified live
  (redirects to Google Cloud Storage, returns valid TFLite bytes);
  `ffmpeg` is already in the Dockerfile's apt packages (needed for RTSP)
- MQTT payload fields verified field-by-field against BirdNET-Go's
  published format (`CommonName`, `ScientificName`, `Confidence`,
  `Date`, `Time`, `Source`); connection pattern (retained messages,
  Supervisor broker lookup) matches the BirdNET-Go HA app approach
- RTSP plumbing (2026-10-12): ffmpeg command builder verified
  (audio-only `-allowed_media_types audio` first, `-vn` fallback —
  same as BirdNET-Go); Frigate go2rtc URL builder verified
  (`rtsp://<host>:8554/<camera>`); source selection logic tested
  (mic default / rtsp / frigate single+multi / frigate priority /
  fatal-on-missing-config paths); live pipe test with ffmpeg-generated
  audio confirmed exact 3 s float32 windows, correct tone content, clean
  EOF → reconnect-signal, and restart

**NOT tested — do not assume these work yet:**
- Docker image build (no Docker in this environment)
- Install on a real Home Assistant host
- Microphone capture on HA hardware
- RTSP against a real camera or Frigate instance (no network/hardware here)
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
