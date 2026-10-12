#!/bin/bash
# Manual Android build for Animal Identify (no Gradle): deps -> AAPT2 -> kotlinc -> D8 -> zipalign -> apksigner
# Modeled on the proven FamilyChat pipeline (~/workspace/family-chat/build.sh).
set -e

export JAVA_HOME=~/jdk
export PATH=$JAVA_HOME/bin:$PATH
mkdir -p ~/workspace/animal-identify/android/build/tmp
export _JAVA_OPTIONS="-Djava.io.tmpdir=$HOME/workspace/animal-identify/android/build/tmp"

ROOT=~/workspace/animal-identify/android
SRC=$ROOT/app/src/main
BUILD=$ROOT/build
SDK=~/Android/Sdk
AAPT2=$SDK/build-tools/37.0.0/aapt2
ZIPALIGN=$SDK/build-tools/37.0.0/zipalign
ANDROID_JAR=$SDK/platforms/android-36/android.jar
D8_JAR=$SDK/build-tools/37.0.0/lib/d8.jar
APKSIGNER_JAR=$SDK/build-tools/37.0.0/lib/apksigner.jar
KOTLINC=~/kotlinc/bin/kotlinc
STDLIB=~/maven-repo/org/jetbrains/kotlin/kotlin-stdlib/1.9.24/kotlin-stdlib-1.9.24.jar
APP_ID=com.animalidentify.app
KEYSTORE=$ROOT/animalidentify.keystore
APK_NAME=AnimalIdentify-v1.apk
DEPS_JSON=$ROOT/deps/deps.json
# Keep the APK lean: real phones are arm64 (Kris's Pixel 7).
ABIS="arm64-v8a armeabi-v7a"

mkdir -p "$BUILD"/{compiled_res,classes,dex,gen}

echo "=== 1. Dependencies (idempotent) ==="
python3 "$ROOT/tools/fetch_deps.py" > "$BUILD/deps.log" 2>&1
tail -3 "$BUILD/deps.log"

echo "=== 2. AAPT2 compile ==="
rm -rf "$BUILD/compiled_res" && mkdir -p "$BUILD/compiled_res"
"$AAPT2" compile --dir "$SRC/res" -o "$BUILD/compiled_res/app-res.zip"

echo "=== 3. AAPT2 link ==="
cp "$SRC/AndroidManifest.xml" "$BUILD/AndroidManifest.xml"
python3 - "$BUILD/AndroidManifest.xml" "$APP_ID" <<'EOF'
import sys
path, app_id = sys.argv[1], sys.argv[2]
with open(path) as f:
    content = f.read()
if 'package="' not in content:
    content = content.replace("<manifest ", '<manifest package="%s" ' % app_id, 1)
with open(path, "w") as f:
    f.write(content)
print("manifest package ok")
EOF
rm -rf "$BUILD/gen" && mkdir -p "$BUILD/gen"
rm -f "$BUILD/base.apk"
"$AAPT2" link -o "$BUILD/base.apk" \
  -I "$ANDROID_JAR" \
  --manifest "$BUILD/AndroidManifest.xml" \
  --min-sdk-version 26 --target-sdk-version 37 \
  --version-code 1 --version-name "1.0" \
  --auto-add-overlay \
  --java "$BUILD/gen" \
  "$BUILD/compiled_res/app-res.zip" \
  -A "$SRC/assets"
echo "base.apk: $(du -h "$BUILD/base.apk" | cut -f1)"

echo "=== 4. Compile R.java ==="
R_JAVA=$(find "$BUILD/gen" -name 'R.java')
javac -nowarn -encoding UTF-8 -cp "$ANDROID_JAR" -d "$BUILD/classes" $R_JAVA 2>&1 | tail -3
echo "R classes: $(find "$BUILD/classes" -name 'R*.class' | wc -l)"

echo "=== 5. kotlinc ==="
CP_JARS=$(python3 -c "import json; print(':'.join(json.load(open('$DEPS_JSON'))['jars']))")
CP="$BUILD/classes:$CP_JARS:$STDLIB:$ANDROID_JAR"
SOURCES=$(find "$SRC/kotlin" -name "*.kt" | tr '\n' ' ')
echo "sources: $(echo "$SOURCES" | wc -w)"
"$KOTLINC" -cp "$CP" -jvm-target 17 -d "$BUILD/classes" $SOURCES > "$BUILD/kotlinc.log" 2>&1
echo "kotlinc exit: $?"
tail -5 "$BUILD/kotlinc.log"

echo "=== 6. Merge dep jars + D8 ==="
python3 - "$DEPS_JSON" "$STDLIB" "$BUILD" <<'EOF'
import json, sys, zipfile, os
deps = json.load(open(sys.argv[1])); stdlib = sys.argv[2]; build = sys.argv[3]
seen = set()
out = os.path.join(build, "merged-deps.jar")
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for jp in deps["jars"] + [stdlib]:
        with zipfile.ZipFile(jp) as zin:
            for n in zin.namelist():
                if n in seen or n.startswith("META-INF/") or not n.endswith(".class"):
                    continue
                seen.add(n)
                z.writestr(n, zin.read(n))
print("merged classes:", len(seen))
EOF
"$JAVA_HOME/bin/jar" cf "$BUILD/app-classes.jar" -C "$BUILD/classes" .
rm -rf "$BUILD/dex" && mkdir -p "$BUILD/dex"
java -Xmx1024m -cp "$D8_JAR" com.android.tools.r8.D8 \
  --lib "$ANDROID_JAR" \
  --min-api 26 \
  --output "$BUILD/dex" \
  "$BUILD/app-classes.jar" "$BUILD/merged-deps.jar" > "$BUILD/d8.log" 2>&1
echo "d8 exit: $?"
tail -3 "$BUILD/d8.log"
echo "dex: $(ls "$BUILD/dex")"

echo "=== 7. Package APK (dex) ==="
python3 - "$BUILD/base.apk" "$BUILD/dex" <<'EOF'
import sys, os, zipfile
apk, dexdir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(apk, 'a', zipfile.ZIP_STORED) as z:
    existing = set(z.namelist())
    for f in sorted(os.listdir(dexdir)):
        if f.startswith("classes") and f.endswith(".dex") and f not in existing:
            z.write(os.path.join(dexdir, f), f)
            print("added", f)
EOF

echo "=== 7b. Inject TFLite native libs (arm64-v8a + armeabi-v7a, deflate-9) ==="
python3 - "$DEPS_JSON" "$BUILD/base.apk" "$ABIS" <<'EOF'
import sys, json, zipfile
deps = json.load(open(sys.argv[1])); apk = sys.argv[2]; abis = sys.argv[3].split()
aar = deps["libs"][0]["aar_file"]
with zipfile.ZipFile(aar) as za:
    sos = {n: za.read(n) for n in za.namelist()
           if n.startswith("jni/") and n.endswith(".so")}
with zipfile.ZipFile(apk, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    existing = set(z.namelist())
    for abi in abis:
        for src, data in sos.items():
            if src.startswith("jni/%s/" % abi):
                name = "lib/%s/%s" % (abi, src.split("/")[-1])
                if name not in existing:
                    z.writestr(name, data)
                    print("injected %s (%d bytes)" % (name, len(data)))
EOF

echo "=== 8. zipalign + sign (persistent keystore) ==="
if [ ! -f "$KEYSTORE" ]; then
  echo "creating ONE persistent keystore (reused for all future Animal Identify builds)"
  keytool -genkeypair -keystore "$KEYSTORE" -storepass animalidentify \
    -keypass animalidentify -keyalg RSA -keysize 2048 -validity 10950 \
    -dname "CN=AnimalIdentify" -alias animalidentify 2>/dev/null
else
  echo "reusing existing keystore: $KEYSTORE"
fi
"$ZIPALIGN" -f -p 4 "$BUILD/base.apk" "$BUILD/aligned.apk"
java -cp "$APKSIGNER_JAR" com.android.apksigner.ApkSignerTool sign \
  --ks "$KEYSTORE" --ks-pass pass:animalidentify \
  --out "$BUILD/$APK_NAME" "$BUILD/aligned.apk"
java -cp "$APKSIGNER_JAR" com.android.apksigner.ApkSignerTool verify --print-certs "$BUILD/$APK_NAME" | head -4
cp "$BUILD/$APK_NAME" "$ROOT/$APK_NAME"
echo "APK: $(du -h "$ROOT/$APK_NAME" | cut -f1) -> $ROOT/$APK_NAME"
