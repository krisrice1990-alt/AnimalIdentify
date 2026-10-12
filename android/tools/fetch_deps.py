#!/usr/bin/env python3
"""Fetch Animal Identify dependencies into deps/ and emit deps.json.

Dependencies (all from Maven Central, reachable through the proxy):
  - org.tensorflow:tensorflow-lite:2.16.1 (AAR: classes.jar + jni .so)

deps.json format mirrors the FamilyChat pipeline:
  {"jars": [...], "libs": [...], "extra_packages": [...]}
"""
import json, os, urllib.request, zipfile

ROOT = os.path.expanduser("~/workspace/animal-identify/android")
DEPS = os.path.join(ROOT, "deps")
os.makedirs(DEPS, exist_ok=True)
MC = "https://repo1.maven.org/maven2"

WANTS = [
    # (group_path, artifact, version, ext)
    # 2.15.0: last monolithic API (2.16+ splits InterpreterApi into a
    # separate artifact that isn't published as a standalone jar).
    ("org/tensorflow", "tensorflow-lite", "2.15.0", "aar"),
]

def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("cached:", os.path.basename(dest))
        return dest
    print("fetch:", url)
    req = urllib.request.Request(url, headers={"User-Agent": "animalidentify-build/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print("saved:", dest, os.path.getsize(dest))
    return dest

jars, libs = [], []
for gpath, art, ver, ext in WANTS:
    fname = f"{art}-{ver}.{ext}"
    url = f"{MC}/{gpath}/{art}/{ver}/{fname}"
    dest = os.path.join(DEPS, fname)
    fetch(url, dest)
    if ext == "aar":
        with zipfile.ZipFile(dest) as z:
            cj = os.path.join(DEPS, f"{art}-{ver}-classes.jar")
            if not os.path.exists(cj):
                with open(cj, "wb") as f:
                    f.write(z.read("classes.jar"))
                print("extracted classes.jar ->", cj)
        safe = f"{art}-{ver}"
        libs.append({"safe": safe, "aar_file": dest, "res_dir": None})
        jars.append(cj)
    else:
        jars.append(dest)

out = {"jars": jars, "libs": libs, "extra_packages": []}
# Add the InterpreterApi compile/runtime stub (the real tensorflow-lite-api
# artifact is referenced by the POM but not published; the AAR's Interpreter
# needs the interface to exist).
stub = os.path.join(DEPS, "tflite-api-stub.jar")
if os.path.exists(stub):
    out["jars"].append(stub)
    print("added stub:", stub)
with open(os.path.join(DEPS, "deps.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote deps.json")
for j in jars:
    print(" jar:", j, os.path.getsize(j))
