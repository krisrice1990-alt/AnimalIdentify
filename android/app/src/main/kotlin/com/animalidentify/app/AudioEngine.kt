package com.animalidentify.app

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

/**
 * Captures 48kHz mono 16-bit PCM in 3-second windows on a background thread.
 * Calls [onWindow] for each full window; calls [onError] on fatal errors.
 */
class AudioEngine(
    private val onWindow: (ShortArray) -> Unit,
    private val onError: (String) -> Unit
) {
    private val sampleRate = ModelConfig.SAMPLE_RATE_IN
    private val windowSamples = sampleRate * ModelConfig.WINDOW_S

    @Volatile
    private var running = false
    private var thread: Thread? = null
    private var recorder: AudioRecord? = null

    fun start(): Boolean {
        if (running) return true
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            onError("Microphone not available on this device.")
            return false
        }
        return try {
            val rec = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuf * 4, windowSamples * 2)
            )
            if (rec.state != AudioRecord.STATE_INITIALIZED) {
                onError("Could not open the microphone.")
                rec.release()
                return false
            }
            recorder = rec
            running = true
            thread = Thread(::loop, "AnimalIdentify-Audio").apply { start() }
            true
        } catch (e: SecurityException) {
            onError("Microphone permission was denied.")
            false
        } catch (e: Exception) {
            onError("Could not start recording: ${e.message}")
            false
        }
    }

    private fun loop() {
        val rec = recorder ?: return
        val window = ShortArray(windowSamples)
        var filled = 0
        try {
            rec.startRecording()
            val chunk = ShortArray(8192)
            while (running) {
                val n = rec.read(chunk, 0, chunk.size)
                if (n < 0) {
                    onError("Microphone read error ($n).")
                    break
                }
                if (n == 0) continue
                var off = 0
                while (off < n && running) {
                    val take = minOf(n - off, windowSamples - filled)
                    System.arraycopy(chunk, off, window, filled, take)
                    filled += take
                    off += take
                    if (filled >= windowSamples) {
                        filled = 0
                        onWindow(window.copyOf())
                    }
                }
            }
        } catch (e: Exception) {
            if (running) onError("Recording stopped: ${e.message}")
        } finally {
            try { rec.stop() } catch (_: Exception) {}
            try { rec.release() } catch (_: Exception) {}
            recorder = null
        }
    }

    fun stop() {
        running = false
        try { thread?.join(2000) } catch (_: Exception) {}
        thread = null
    }
}
