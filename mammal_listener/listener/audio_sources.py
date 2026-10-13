"""Audio input sources for Mammal Listener.

Two sources, one interface — both deliver 48 kHz mono float32 samples:

  MicSource  - local microphone via sounddevice (the original path).
  RtspSource - network audio via ffmpeg (IP cameras, Frigate go2rtc).

Interface:
  start()                      begin capture (may raise SourceLost)
  read_window(n) -> np.ndarray  block until n samples are available
                                (raises SourceLost if the stream dies)
  reconnect()                  wait with backoff, then start() again
  stop()                       release resources
  describe() -> str             human-readable label for logs

RtspSource requests audio-only from the
camera first (`-allowed_media_types audio`, so no video decoder slot is
wasted on the camera), and fall back to the full stream with `-vn` when
the audio-only handshake never delivers audio. After two consecutive
audio-only failures the fallback becomes permanent for that stream.
"""

import select
import subprocess
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000

# How long to wait for the first audio bytes before declaring the
# audio-only handshake a failure (seconds).
AUDIO_ONLY_TIMEOUT_S = 10
# If no bytes arrive for this long on a live process, treat as stalled.
STALL_TIMEOUT_S = 20


class SourceLost(Exception):
    """Raised when an audio source dies or stalls; caller should reconnect."""


def build_ffmpeg_cmd(url, audio_only=True):
    """Build the ffmpeg command that turns an RTSP URL into raw audio.

    Output is always 48 kHz mono float32 little-endian on stdout,
    regardless of what the camera sends.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
    ]
    if audio_only:
        cmd += ["-allowed_media_types", "audio"]
    cmd += [
        "-i",
        url,
        "-vn",  # never decode video; we only want the audio track
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    return cmd


def build_frigate_url(host, camera):
    """Frigate restreams cameras via go2rtc at rtsp://<host>:8554/<camera>."""
    return f"rtsp://{host}:8554/{camera}"


class MicSource:
    """Local microphone capture (original Mammal Listener path)."""

    def __init__(self, sample_rate=SAMPLE_RATE):
        self._sr = sample_rate
        self._q = []
        self._stream = None

    def describe(self):
        return "microphone"

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"audio status: {status}", flush=True)
        self._q.append(indata[:, 0].copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def read_window(self, n):
        buf = []
        need = n
        while need > 0:
            while self._q and need > 0:
                chunk = self._q.pop(0)
                take = min(len(chunk), need)
                buf.append(chunk[:take])
                need -= take
                if take < len(chunk):
                    self._q.insert(0, chunk[take:])
            if need > 0:
                time.sleep(0.05)
        return (
            np.concatenate(buf).astype(np.float32)
            if buf
            else np.zeros(n, dtype=np.float32)
        )

    def reconnect(self):
        # A local mic rarely "reconnects"; just restart the stream.
        self.stop()
        time.sleep(2)
        self.start()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class RtspSource:
    """Network audio from an RTSP URL (IP camera or Frigate go2rtc)."""

    def __init__(self, url, label="rtsp", _cmd_override=None):
        self.url = url
        self.label = label
        self._cmd_override = _cmd_override  # test hook: bypass URL entirely
        self._proc = None
        self._pending = bytearray()
        self._last_data_ts = 0.0
        self._prefer_audio_only = True
        self._audio_only_failures = 0
        self._consec_failures = 0

    def describe(self):
        return f"rtsp:{self.label}"

    # -- process management -------------------------------------------------
    def _spawn(self, audio_only):
        if self._cmd_override is not None:
            cmd = self._cmd_override
        else:
            cmd = build_ffmpeg_cmd(self.url, audio_only=audio_only)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=4 * SAMPLE_RATE * 4,  # ~4 s of float32 mono
        )
        self._pending = bytearray()
        self._last_data_ts = time.time()

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def _wait_for_audio(self, timeout):
        """True if any audio bytes arrive within `timeout` seconds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                return False
            r, _, _ = select.select([self._proc.stdout], [], [], 1.0)
            if r:
                data = self._proc.stdout.read(4096)
                if data:
                    self._pending.extend(data)
                    self._last_data_ts = time.time()
                    return True
                return False  # EOF: process is dying
        return False

    # -- public interface ---------------------------------------------------
    def start(self):
        use_audio_only = self._prefer_audio_only
        self._spawn(use_audio_only)
        if not self._wait_for_audio(AUDIO_ONLY_TIMEOUT_S):
            self._kill()
            if use_audio_only:
                # Behaviour: audio-only first, fall back to the
                # full stream (video discarded) when the camera can't do
                # an audio-only handshake.
                self._audio_only_failures += 1
                if self._audio_only_failures >= 2:
                    self._prefer_audio_only = False
                print(
                    f"[{self.label}] audio-only RTSP handshake failed, "
                    "falling back to full stream",
                    flush=True,
                )
                self._spawn(False)
                if not self._wait_for_audio(AUDIO_ONLY_TIMEOUT_S + 5):
                    self._kill()
                    raise SourceLost(f"no audio from {self.describe()}")
            else:
                raise SourceLost(f"no audio from {self.describe()}")
        self._consec_failures = 0
        print(f"[{self.label}] audio flowing from {self.describe()}", flush=True)

    def _read_bytes(self, nbytes, timeout=5.0):
        """Block until exactly nbytes are buffered; raises SourceLost."""
        while len(self._pending) < nbytes:
            if self._proc is None or self._proc.poll() is not None:
                # Drain whatever is left, then report the death.
                if self._pending:
                    break
                raise SourceLost(f"ffmpeg exited for {self.describe()}")
            if time.time() - self._last_data_ts > STALL_TIMEOUT_S:
                raise SourceLost(f"stream stalled for {self.describe()}")
            r, _, _ = select.select([self._proc.stdout], [], [], timeout)
            if not r:
                continue
            data = self._proc.stdout.read(nbytes - len(self._pending))
            if not data:
                raise SourceLost(f"EOF on ffmpeg stdout for {self.describe()}")
            self._pending.extend(data)
            self._last_data_ts = time.time()
        out = bytes(self._pending[:nbytes])
        del self._pending[:nbytes]
        return out

    def read_window(self, n):
        raw = self._read_bytes(n * 4)
        return np.frombuffer(raw, dtype=np.float32).astype(np.float32, copy=True)

    def reconnect(self):
        self._kill()
        self._consec_failures += 1
        delay = min(60, 5 * (2 ** min(self._consec_failures - 1, 3)))
        print(
            f"[{self.label}] reconnecting in {delay}s "
            f"(attempt {self._consec_failures})",
            flush=True,
        )
        time.sleep(delay)
        self.start()

    def stop(self):
        self._kill()
