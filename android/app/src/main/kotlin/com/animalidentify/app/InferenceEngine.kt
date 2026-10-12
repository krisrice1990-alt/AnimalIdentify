package com.animalidentify.app

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * On-device inference pipeline, mirroring the proven Python pipeline:
 *  3s @ 48kHz mono -> energy gate (-45 dBFS) -> resample to 16kHz ->
 *  YAMNet (15600-sample windows, hop 7680) -> mean-pool -> [1, 521] ->
 *  mammal head -> softmax -> top class + confidence.
 *
 * Model config is centralized here so the real 6-class model drops in later.
 */
object ModelConfig {
    const val YAMNET_ASSET = "yamnet.tflite"
    const val HEAD_ASSET = "mammal-head.tflite"
    const val MODEL_VERSION = "mammal-head-draft-v1"

    /** Draft model classes. Replace with 6-class list when the real model ships. */
    val CLASSES = listOf("wolf", "fox")
    val CLASS_DISPLAY = mapOf("wolf" to "Wolf", "fox" to "Fox")

    const val SAMPLE_RATE_IN = 48000
    const val SAMPLE_RATE_YAMNET = 16000
    const val WINDOW_S = 3
    const val YAMNET_WIN = 15600
    const val YAMNET_HOP = 7680
    const val ENERGY_THRESHOLD_DB = -45.0
    const val CONFIDENCE_THRESHOLD = 0.6
}

data class DetectionResult(
    val label: String,
    val displayName: String,
    val confidence: Float,
    val timestampMs: Long
)

class InferenceEngine(context: Context) {

    private val yamnet: Interpreter?
    private val head: Interpreter?
    val isReady: Boolean

    init {
        var y: Interpreter? = null
        var h: Interpreter? = null
        try {
            y = Interpreter(loadAsset(context, ModelConfig.YAMNET_ASSET))
            h = Interpreter(loadAsset(context, ModelConfig.HEAD_ASSET))
        } catch (e: Exception) {
            try { y?.close() } catch (_: Exception) {}
            try { h?.close() } catch (_: Exception) {}
            y = null
            h = null
        }
        yamnet = y
        head = h
        isReady = yamnet != null && head != null
    }

    private fun loadAsset(context: Context, name: String): ByteBuffer {
        val afd = context.assets.openFd(name)
        val fis = FileInputStream(afd.fileDescriptor)
        val buf = fis.channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
        )
        buf.order(ByteOrder.nativeOrder())
        return buf
    }

    /** Peak dBFS of a 16-bit PCM window. */
    private fun dbfs(samples: ShortArray): Double {
        var sum = 0.0
        for (s in samples) {
            val v = s / 32768.0
            sum += v * v
        }
        val rms = sqrt(sum / samples.size)
        return 20.0 * log10(rms + 1e-12)
    }

    /** Decimate 48kHz -> 16kHz (integer ratio 3:1). Output in [-1, 1] float. */
    private fun resample48to16(samples: ShortArray): FloatArray {
        val outLen = samples.size / 3
        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            out[i] = samples[i * 3] / 32768f
        }
        return out
    }

    /**
     * Run the full pipeline on one 3s @ 48kHz mono window.
     * Returns null when: energy gate fails, confidence too low, or on error.
     */
    fun analyze(window48k: ShortArray): DetectionResult? {
        val y = yamnet ?: return null
        val h = head ?: return null
        try {
            // 1. Energy gate — silence costs nothing.
            if (dbfs(window48k) < ModelConfig.ENERGY_THRESHOLD_DB) return null

            // 2. Resample to 16kHz.
            val wave16k = resample48to16(window48k)

            // 3. YAMNet framing: 15600-sample windows, hop 7680.
            val nFrames = (wave16k.size - ModelConfig.YAMNET_WIN) / ModelConfig.YAMNET_HOP + 1
            if (nFrames < 1) return null
            val pooled = FloatArray(521)
            val inBuf = ByteBuffer.allocateDirect(ModelConfig.YAMNET_WIN * 4)
                .order(ByteOrder.nativeOrder())
            val outBuf = ByteBuffer.allocateDirect(521 * 4)
                .order(ByteOrder.nativeOrder())

            for (f in 0 until nFrames) {
                val start = f * ModelConfig.YAMNET_HOP
                inBuf.rewind()
                for (i in 0 until ModelConfig.YAMNET_WIN) {
                    inBuf.putFloat(wave16k[start + i])
                }
                inBuf.rewind()
                outBuf.rewind()
                y.run(inBuf, outBuf)
                outBuf.rewind()
                for (i in 0 until 521) {
                    pooled[i] += outBuf.float
                }
            }
            for (i in 0 until 521) {
                pooled[i] = pooled[i] / nFrames.toFloat()
            }

            // 4. Mammal head: [1, 521] -> [1, nClasses] softmax.
            val headIn = ByteBuffer.allocateDirect(521 * 4)
                .order(ByteOrder.nativeOrder())
            for (v in pooled) headIn.putFloat(v)
            headIn.rewind()
            val nClasses = ModelConfig.CLASSES.size
            val headOut = ByteBuffer.allocateDirect(nClasses * 4)
                .order(ByteOrder.nativeOrder())
            h.run(headIn, headOut)
            headOut.rewind()

            var topIdx = 0
            var topConf = headOut.float
            for (i in 1 until nClasses) {
                val c = headOut.float
                if (c > topConf) {
                    topConf = c
                    topIdx = i
                }
            }

            // 5. Confidence gate.
            if (topConf < ModelConfig.CONFIDENCE_THRESHOLD) return null

            val label = ModelConfig.CLASSES[topIdx]
            return DetectionResult(
                label = label,
                displayName = ModelConfig.CLASS_DISPLAY[label] ?: label,
                confidence = topConf,
                timestampMs = System.currentTimeMillis()
            )
        } catch (e: Exception) {
            return null
        }
    }

    fun close() {
        try { yamnet?.close() } catch (_: Exception) {}
        try { head?.close() } catch (_: Exception) {}
    }
}
