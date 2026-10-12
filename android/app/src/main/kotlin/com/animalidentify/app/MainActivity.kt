package com.animalidentify.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import java.util.concurrent.Executors

class MainActivity : Activity() {

    private lateinit var recordButton: Button
    private lateinit var historyButton: Button
    private lateinit var statusText: TextView
    private lateinit var resultText: TextView
    private lateinit var confidenceText: TextView

    private var inference: InferenceEngine? = null
    private var audio: AudioEngine? = null
    private var db: DetectionDb? = null
    private val bg = Executors.newSingleThreadExecutor()

    @Volatile
    private var listening = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recordButton = findViewById(R.id.recordButton)
        historyButton = findViewById(R.id.historyButton)
        statusText = findViewById(R.id.statusText)
        resultText = findViewById(R.id.resultText)
        confidenceText = findViewById(R.id.confidenceText)

        db = DetectionDb(this)

        // Load models off the UI thread.
        statusText.text = "Loading AI models…"
        bg.execute {
            val eng = InferenceEngine(this)
            runOnUiThread {
                if (eng.isReady) {
                    inference = eng
                    statusText.text = getString(R.string.tap_to_listen)
                    recordButton.isEnabled = true
                } else {
                    statusText.text = getString(R.string.model_missing)
                    recordButton.isEnabled = false
                    Toast.makeText(
                        this, R.string.model_missing, Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
        recordButton.isEnabled = false

        recordButton.setOnClickListener {
            if (listening) stopListening() else checkPermissionAndStart()
        }
        historyButton.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }
    }

    private fun checkPermissionAndStart() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startListening()
        } else {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 100)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == 100 &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startListening()
        } else {
            Toast.makeText(
                this, R.string.mic_permission_denied, Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun startListening() {
        val eng = inference ?: run {
            Toast.makeText(this, R.string.model_missing, Toast.LENGTH_LONG).show()
            return
        }
        listening = true
        runOnUiThread {
            recordButton.text = getString(R.string.stop)
            statusText.text = getString(R.string.listening)
            resultText.text = ""
            confidenceText.text = ""
        }
        audio = AudioEngine(
            onWindow = { window ->
                // Inference already runs off the audio thread; keep it serial.
                val result = try {
                    eng.analyze(window)
                } catch (e: Exception) {
                    null
                }
                if (result != null) {
                    db?.insert(
                        result.timestampMs,
                        ModelConfig.WINDOW_S,
                        result.label,
                        result.displayName,
                        result.confidence,
                        ModelConfig.MODEL_VERSION
                    )
                    runOnUiThread {
                        resultText.text = result.displayName
                        confidenceText.text =
                            "${(result.confidence * 100).toInt()}% ${getString(R.string.confidence)}"
                    }
                }
            },
            onError = { msg ->
                runOnUiThread {
                    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                    if (listening) stopListening()
                }
            }
        )
        if (audio?.start() != true) {
            listening = false
            runOnUiThread {
                recordButton.text = getString(R.string.tap_to_listen)
                statusText.text = getString(R.string.tap_to_listen)
            }
        }
    }

    private fun stopListening() {
        listening = false
        audio?.stop()
        audio = null
        recordButton.text = getString(R.string.tap_to_listen)
        statusText.text = getString(R.string.tap_to_listen)
    }

    override fun onPause() {
        super.onPause()
        if (listening) stopListening()
    }

    override fun onDestroy() {
        super.onDestroy()
        audio?.stop()
        bg.execute {
            inference?.close()
            db?.close()
        }
        bg.shutdown()
    }
}
