ARG BUILD_FROM
FROM ${BUILD_FROM}

# PortAudio for microphone capture, then the Python detector.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY listener/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YAMNet feature extractor (TF Hub) + the trained 4-class mammal head
# (wolf/coyote/moose/dog), both bundled into the image so the add-on
# works out of the box with no manual model copying.
ARG YAMNET_URL=https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite
RUN mkdir -p /app/models \
    && python3 -c "import urllib.request; urllib.request.urlretrieve('${YAMNET_URL}', '/app/models/yamnet.tflite')" \
    && ls -la /app/models/
COPY models/mammal-head.tflite /app/models/mammal-head.tflite

COPY listener/ /app/listener/
COPY rootfs/ /

CMD ["/etc/services.d/mammal_listener/run"]
