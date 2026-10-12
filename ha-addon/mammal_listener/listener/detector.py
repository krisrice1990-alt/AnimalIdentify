#!/usr/bin/env python3
"""Mammal Listener: low-CPU on-device animal detection for Home Assistant.

Pipeline (matches mammal-model/INTERFACE.md):
  mic 48 kHz mono -> 3 s windows -> energy gate (cheap) ->
  resample 16 kHz -> YAMNet frames -> mean-pooled 521 posteriors ->
  mammal-head.tflite -> softmax over [wolf, coyote, moose, bear, dog, background]

Only windows that pass the energy gate reach YAMNet, so a quiet porch costs
almost no CPU. Detections publish over MQTT with HA auto-discovery.

Config: /data/options.json (written by the HA Supervisor from config.yaml).
Model:  /share/mammal-head.tflite (copied there after training; NOT shipped).
"""
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
from scipy.signal import resample_poly

CLASSES = ["wolf", "coyote", "moose", "bear", "dog", "background"]
SR = 48000
WIN_S = 3.0
WIN_N = int(SR * WIN_S)

# BirdNET-Go-compatible species names (dual-dialect payload).
SPECIES_INFO = {
    "wolf": ("Gray Wolf", "Canis lupus"),
    "coyote": ("Coyote", "Canis latrans"),
    "moose": ("Moose", "Alces alces"),
    "bear": ("Bear", "Ursus americanus"),
    "dog": ("Dog", "Canis lupus familiaris"),
    "background": ("Background", ""),
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
    """Pull MQTT broker credentials from the HA Supervisor, BirdNET-Go style.

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


def main():
    opt = load_options()

    head_path = "/share/mammal-head.tflite"
    if not os.path.exists(head_path):
        print(
            "FATAL: /share/mammal-head.tflite not found. Train the model first "
            "(mammal-model/README.md) and copy mammal-head.tflite to the "
            "add-on's /share directory, then restart.",
            flush=True,
        )
        sys.exit(1)

    print("Loading YAMNet extractor...", flush=True)
    yamnet = load_tflite("/app/models/yamnet.tflite")
    print("Loading mammal head...", flush=True)
    head = load_tflite(head_path)

    base = opt["base_topic"].strip().strip("/")

    # MQTT connection: BirdNET-Go style. When mqtt_auto_config is on (default),
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
            "add-on installed, or set mqtt_host manually.",
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
        "model": "mammal-head v1",
    }
    for cls in CLASSES:
        if cls in ("dog", "background"):
            continue
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
    print("Discovery published. Listening...", flush=True)

    energy_db = float(opt.get("energy_threshold_db", -45))
    conf_thr = float(opt.get("confidence_threshold", 0.6))
    cooldown_s = float(opt.get("cooldown_minutes", 10)) * 60.0
    last_fired = {c: 0.0 for c in CLASSES}

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"audio status: {status}", flush=True)
        window_q.append(indata[:, 0].copy())

    window_q = []
    stream = sd.InputStream(
        samplerate=SR, channels=1, dtype="float32", callback=audio_callback
    )
    stream.start()
    try:
        while True:
            # Assemble one 3 s window.
            buf = []
            need = WIN_N
            while need > 0:
                time.sleep(0.05)
                while window_q and need > 0:
                    chunk = window_q.pop(0)
                    take = min(len(chunk), need)
                    buf.append(chunk[:take])
                    need -= take
                    if take < len(chunk):
                        window_q.insert(0, chunk[take:])
            clip = np.concatenate(buf) if buf else np.zeros(WIN_N, dtype=np.float32)

            # Energy gate: silence costs nothing beyond this line.
            if dbfs(clip) < energy_db:
                continue

            # 48 kHz -> 16 kHz, then YAMNet frames.
            clip16 = resample_poly(clip, 1, 3).astype(np.float32)
            frames = []
            for start in range(0, len(clip16) - YAMNET_WIN + 1, YAMNET_HOP):
                frames.append(clip16[start : start + YAMNET_WIN])
            if not frames:
                continue
            posts = []
            for fr in frames:
                out = run_tflite(yamnet, fr.reshape(1, -1))
                posts.append(out.reshape(-1))
            pooled = np.mean(np.stack(posts), axis=0, keepdims=True)  # (1, 521)

            scores = run_tflite(head, pooled).reshape(-1)  # (6,) softmax
            top = int(np.argmax(scores))
            now = time.time()
            if scores[top] >= conf_thr and now - last_fired[CLASSES[top]] >= cooldown_s:
                cls = CLASSES[top]
                last_fired[cls] = now
                common, scientific = SPECIES_INFO[cls]
                conf = round(float(scores[top]), 4)
                dt = time.localtime(now)
                # Dual-dialect payload: PascalCase = BirdNET-Go, snake_case =
                # BirdNET-Pi. Any HA setup that understands BirdNET-Go
                # understands this with zero changes.
                payload = {
                    "Time": time.strftime("%H:%M:%S", dt),
                    "Date": time.strftime("%Y-%m-%d", dt),
                    "CommonName": common,
                    "ScientificName": scientific,
                    "Confidence": conf,
                    "Source": "animal-identify",
                    "time": time.strftime("%H:%M:%S", dt),
                    "date": time.strftime("%Y-%m-%d", dt),
                    "common_name": common,
                    "scientific_name": scientific,
                    "confidence_score": conf,
                }
                if cls not in ("dog", "background"):
                    client.publish(
                        f"{base}/detections/{cls}",
                        json.dumps(payload),
                        qos=1,
                        retain=True,
                    )
                    client.publish(
                        f"{base}/detections/last", json.dumps(payload), qos=1, retain=True
                    )
                print(f"detection: {payload}", flush=True)
    finally:
        stream.stop()
        client.publish(f"{base}/status", "offline", qos=1, retain=True)
        client.loop_stop()


if __name__ == "__main__":
    main()
