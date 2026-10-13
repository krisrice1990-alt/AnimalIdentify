#!/usr/bin/env python3
"""Mammal Listener: low-CPU on-device animal detection for Home Assistant.

Pipeline (matches mammal-model/INTERFACE.md):
  audio 48 kHz mono -> 3 s windows -> energy gate (cheap) ->
  resample 16 kHz -> YAMNet frames -> mean-pooled 521 posteriors ->
  mammal-head.tflite -> softmax over [wolf, coyote, moose, dog]

Audio sources (see listener/audio_sources.py):
  - local microphone (default, `audio_source: mic`)
  - any RTSP URL, e.g. an IP camera (`audio_source: rtsp`,
    `rtsp_url: rtsp://...`) — ffmpeg pulls audio-only first
  - Frigate cameras via go2rtc restream (`frigate_enabled: true`)

Only windows that pass the energy gate reach YAMNet, so a quiet porch costs
almost no CPU. Detections publish over MQTT with HA auto-discovery.

Config: /data/options.json (written by the HA Supervisor from config.yaml).
Model:  bundled at /app/models/mammal-head.tflite (built into the app
         image). Set the `model_path` option to point at a custom model
         (e.g. /share/mammal-head.tflite) to override it.
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import paho.mqtt.client as mqtt
from scipy.signal import resample_poly

from audio_sources import (
    MicSource,
    RtspSource,
    SourceLost,
    build_frigate_url,
)
import webui

# 4-class model v1: wolf, coyote, moose, dog.
# (Bear dropped — no real recordings exist; fox excluded — unverified.)
# Clean-test accuracy on Kris-verified audio: 66.6%.
CLASSES = ["wolf", "coyote", "moose", "dog"]

# App version (keep in sync with `version:` in config.yaml and
# listener/webui.py). Shown in the bug-report UI diagnostics.
APP_VERSION = "1.0.0"
SR = 48000
WIN_S = 3.0
WIN_N = int(SR * WIN_S)

# Species names for the dual-dialect payload.
SPECIES_INFO = {
    "wolf": ("Gray Wolf", "Canis lupus"),
    "coyote": ("Coyote", "Canis latrans"),
    "moose": ("Moose", "Alces alces"),
    "dog": ("Dog", "Canis lupus familiaris"),
}

# YAMNet framing per INTERFACE.md (16 kHz): 15,600-sample windows, hop 7,680.
YAMNET_WIN = 15600
YAMNET_HOP = 7680


def load_options():
    with open("/data/options.json") as f:
        return json.load(f)


def dbfs(x):
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * math.log10(rms + 1e-12)


def load_tflite(path):
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=1)  # 1 thread: low CPU by design
    it.allocate_tensors()
    return it


def supervisor_mqtt_config():
    """Pull MQTT broker credentials from the HA Supervisor.

    Returns (host, port, username, password, use_tls) or None if the
    Supervisor API is unavailable or no MQTT service is configured.
    """
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return None
    try:
        req = urllib.request.Request(
            "http://supervisor/services/mqtt",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("data", {})
        host = data.get("host")
        if not host:
            return None
        return (
            host,
            int(data.get("port", 1883)),
            data.get("username") or "",
            data.get("password") or "",
            bool(data.get("ssl", False)),
        )
    except Exception as e:
        print(f"Supervisor MQTT lookup failed: {e}", flush=True)
        return None


def run_tflite(it, x):
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    it.set_tensor(inp["index"], x.astype(np.float32))
    it.invoke()
    return it.get_tensor(out["index"])


def resolve_sources(opt):
    """Build the list of (label, AudioSource) from the app options.

    Priority: Frigate cameras > explicit RTSP URL > local microphone.
    """
    if opt.get("frigate_enabled"):
        host = (opt.get("frigate_host") or "").strip()
        if not host:
            print("FATAL: frigate_enabled is true but frigate_host is empty.",
                  flush=True)
            sys.exit(1)
        cams = list(opt.get("frigate_cameras") or [])
        single = (opt.get("frigate_camera") or "").strip()
        if single and single not in cams:
            cams.insert(0, single)
        if not cams:
            print("FATAL: frigate_enabled is true but no frigate_camera / "
                  "frigate_cameras are set.", flush=True)
            sys.exit(1)
        return [
            (cam, RtspSource(build_frigate_url(host, cam), label=f"frigate:{cam}"))
            for cam in cams
        ]
    if (opt.get("audio_source") or "mic") == "rtsp":
        url = (opt.get("rtsp_url") or "").strip()
        if not url:
            print('FATAL: audio_source is "rtsp" but rtsp_url is empty.',
                  flush=True)
            sys.exit(1)
        return [("rtsp", RtspSource(url, label="rtsp"))]
    return [("mic", MicSource())]


def run_source(label, source, yamnet, head, ctx):
    """Capture -> energy gate -> YAMNet -> mammal head -> MQTT, forever.

    One thread runs this per audio source. `ctx` carries the shared MQTT
    client, topic base, thresholds, and per-(source, class) cooldowns.
    """
    client = ctx["client"]
    base = ctx["base"]
    energy_db = ctx["energy_db"]
    conf_thr = ctx["conf_thr"]
    cooldown_s = ctx["cooldown_s"]
    last_fired = ctx["last_fired"]

    # Initial connect: keep retrying with backoff (a camera may not be
    # up yet when the app starts).
    while True:
        try:
            source.start()
            break
        except SourceLost as e:
            print(f"[{label}] {e}", flush=True)
            try:
                source.reconnect()
            except Exception as e2:  # never let a source thread die silently
                print(f"[{label}] reconnect failed: {e2}", flush=True)
    print(f"[{label}] listening via {source.describe()}", flush=True)
    while True:
        try:
            clip = source.read_window(WIN_N)
        except SourceLost as e:
            print(f"[{label}] {e}", flush=True)
            try:
                source.reconnect()
            except Exception as e2:  # never let a source thread die silently
                print(f"[{label}] reconnect failed: {e2}", flush=True)
            continue
        except Exception as e:
            print(f"[{label}] capture error: {e}", flush=True)
            break

        # Energy gate: silence costs nothing beyond this line.
        if dbfs(clip) < energy_db:
            continue

        # 48 kHz -> 16 kHz, then YAMNet frames.
        clip16 = resample_poly(clip, 1, 3).astype(np.float32)
        frames = []
        for start in range(0, len(clip16) - YAMNET_WIN + 1, YAMNET_HOP):
            frames.append(clip16[start:start + YAMNET_WIN])
        if not frames:
            continue
        posts = []
        for fr in frames:
            out = run_tflite(yamnet, fr.reshape(1, -1))
            posts.append(out.reshape(-1))
        pooled = np.mean(np.stack(posts), axis=0, keepdims=True)  # (1, 521)

        scores = run_tflite(head, pooled).reshape(-1)  # (4,) softmax
        top = int(np.argmax(scores))
        now = time.time()
        key = (label, CLASSES[top])
        if scores[top] >= conf_thr and now - last_fired.get(key, 0.0) >= cooldown_s:
            cls = CLASSES[top]
            last_fired[key] = now
            common, scientific = SPECIES_INFO[cls]
            conf = round(float(scores[top]), 4)
            dt = time.localtime(now)
            # Dual-dialect payload: PascalCase and snake_case field names.
            # Any HA setup that understands either dialect
            # understands this with zero changes. "Camera" names the audio
            # source (mic / rtsp / frigate:<camera>).
            payload = {
                "Time": time.strftime("%H:%M:%S", dt),
                "Date": time.strftime("%Y-%m-%d", dt),
                "CommonName": common,
                "ScientificName": scientific,
                "Confidence": conf,
                "Source": "animal-identify",
                "Camera": label,
                "time": time.strftime("%H:%M:%S", dt),
                "date": time.strftime("%Y-%m-%d", dt),
                "common_name": common,
                "scientific_name": scientific,
                "confidence_score": conf,
                "camera": label,
            }
            client.publish(
                f"{base}/detections/{cls}",
                json.dumps(payload),
                qos=1,
                retain=True,
            )
            client.publish(
                f"{base}/detections/last", json.dumps(payload), qos=1, retain=True
            )
            print(f"[{label}] detection: {payload}", flush=True)


def main():
    opt = load_options()

    # Bug-report web UI first: runs in its own daemon thread so it stays
    # reachable even if the detector later fails on MQTT/model config.
    webui.start(opt, version=APP_VERSION)

    # Model: bundled in the image, overridable via the `model_path` option.
    head_path = opt.get("model_path") or "/app/models/mammal-head.tflite"
    if not os.path.exists(head_path):
        print(
            f"FATAL: mammal-head model not found at {head_path}. The app "
            "image bundles it at /app/models/mammal-head.tflite; if you set "
            "model_path manually, make sure that file exists.",
            flush=True,
        )
        sys.exit(1)

    # Resolve audio sources before touching the network: fail fast on a
    # bad rtsp_url / frigate config instead of connecting MQTT first.
    sources = resolve_sources(opt)
    print(f"Audio source(s): {[label for label, _ in sources]}", flush=True)

    # Fail fast if YAMNet is missing (per-thread interpreters load later).
    yamnet_path = "/app/models/yamnet.tflite"
    if not os.path.exists(yamnet_path):
        print(f"FATAL: YAMNet model not found at {yamnet_path}.", flush=True)
        sys.exit(1)

    base = opt["base_topic"].strip().strip("/")

    # MQTT connection. When mqtt_auto_config is on (default),
    # broker credentials come straight from the HA Supervisor — no manual entry.
    # Falls back to the manual mqtt_host/port/user/password options.
    mqtt_host = opt.get("mqtt_host") or ""
    mqtt_port = int(opt.get("mqtt_port") or 1883)
    mqtt_user = opt.get("mqtt_user") or ""
    mqtt_password = opt.get("mqtt_password") or ""
    mqtt_tls = bool(opt.get("mqtt_tls"))
    if opt.get("mqtt_auto_config"):
        auto = supervisor_mqtt_config()
        if auto:
            mqtt_host, mqtt_port, mqtt_user, mqtt_password, mqtt_tls = auto
            print(
                f"MQTT auto-config: broker {mqtt_host}:{mqtt_port} "
                f"(tls={mqtt_tls}) from Supervisor",
                flush=True,
            )
        else:
            print(
                "MQTT auto-config found nothing; using manual options",
                flush=True,
            )
    if not mqtt_host:
        print(
            "FATAL: no MQTT broker. Enable mqtt_auto_config with the Mosquitto "
            "app installed, or set mqtt_host manually.",
            flush=True,
        )
        sys.exit(1)

    client = mqtt.Client(client_id="mammal_listener")
    if mqtt_user:
        client.username_pw_set(mqtt_user, mqtt_password)
    if mqtt_tls:
        client.tls_set()
    client.will_set(f"{base}/status", "offline", qos=1, retain=True)
    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()
    client.publish(f"{base}/status", "online", qos=1, retain=True)

    # HA auto-discovery: one sensor per animal class + a status sensor.
    device = {
        "identifiers": ["mammal_listener"],
        "name": "Mammal Listener",
        "manufacturer": "AnimalIdentify",
        "model": "mammal-head 4-class v1",
    }
    for cls in CLASSES:
        cfg_topic = f"homeassistant/sensor/mammal_listener/{cls}/config"
        client.publish(
            cfg_topic,
            json.dumps(
                {
                    "name": f"Mammal Listener {cls}",
                    "unique_id": f"mammal_listener_{cls}",
                    "state_topic": f"{base}/detections/{cls}",
                    "value_template": "{{ (value_json.Confidence * 100) | round(1) }}",
                    "unit_of_measurement": "%",
                    "json_attributes_topic": f"{base}/detections/{cls}",
                    "device": device,
                }
            ),
            qos=1,
            retain=True,
        )
    print("Discovery published.", flush=True)

    # Each source gets its own thread and its own YAMNet/head
    # interpreters (interpreters are not thread-safe); they share the
    # MQTT client (paho publish is thread-safe).
    ctx = {
        "client": client,
        "base": base,
        "energy_db": float(opt.get("energy_threshold_db", -45)),
        "conf_thr": float(opt.get("confidence_threshold", 0.6)),
        "cooldown_s": float(opt.get("cooldown_minutes", 10)) * 60.0,
        "last_fired": {},  # (source_label, class) -> timestamp
    }

    threads = []
    for label, source in sources:
        # Interpreters are not thread-safe: one pair per source thread.
        src_yamnet = load_tflite("/app/models/yamnet.tflite")
        src_head = load_tflite(head_path)
        t = threading.Thread(
            target=run_source,
            args=(label, source, src_yamnet, src_head, ctx),
            name=f"mammal-src-{label}",
            daemon=True,
        )
        t.start()
        threads.append(t)
    print(f"Discovery published. Listening on {len(sources)} source(s)...",
          flush=True)
    try:
        for t in threads:
            t.join()
    finally:
        for _label, source in sources:
            try:
                source.stop()
            except Exception:
                pass
        client.publish(f"{base}/status", "offline", qos=1, retain=True)
        client.loop_stop()


if __name__ == "__main__":
    main()
