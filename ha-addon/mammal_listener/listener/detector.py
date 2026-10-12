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

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
from scipy.signal import resample_poly

CLASSES = ["wolf", "coyote", "moose", "bear", "dog", "background"]
SR = 48000
WIN_S = 3.0
WIN_N = int(SR * WIN_S)

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
    client = mqtt.Client(client_id="mammal_listener")
    if opt.get("mqtt_user"):
        client.username_pw_set(opt["mqtt_user"], opt.get("mqtt_password") or "")
    if opt.get("mqtt_tls"):
        client.tls_set()
    client.will_set(f"{base}/status", "offline", qos=1, retain=True)
    client.connect(opt["mqtt_host"], int(opt["mqtt_port"]), keepalive=60)
    client.loop_start()
    client.publish(f"{base}/status", "online", qos=1, retain=True)

    # HA auto-discovery: one sensor per animal class + a status sensor.
    device = {
        "identifiers": ["mammal_listener"],
        "name": "Mammal Listener",
        "manufacturer": "FieldListener",
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
                    "value_template": "{{ value_json.confidence }}",
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
                payload = {
                    "species": cls,
                    "confidence": round(float(scores[top]) * 100.0, 1),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "source": "mammal_listener",
                }
                if cls not in ("dog", "background"):
                    client.publish(
                        f"{base}/detections/{cls}", json.dumps(payload), qos=1
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
