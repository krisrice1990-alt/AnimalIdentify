package com.animalidentify.app

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class Recording(
    val id: Long,
    val timestampMs: Long,
    val durationS: Int,
    val topLabel: String,
    val topDisplay: String,
    val confidence: Float,
    val modelVersion: String
)

class DetectionDb(context: Context) :
    SQLiteOpenHelper(context, "animal_identify.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """CREATE TABLE recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                duration_s INTEGER NOT NULL,
                top_label TEXT NOT NULL,
                top_display TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL
            )"""
        )
        db.execSQL("CREATE INDEX idx_ts ON recordings(timestamp_ms DESC)")
    }

    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        // v1: nothing to migrate yet.
    }

    fun insert(
        timestampMs: Long,
        durationS: Int,
        label: String,
        display: String,
        confidence: Float,
        modelVersion: String
    ): Long {
        val cv = ContentValues().apply {
            put("timestamp_ms", timestampMs)
            put("duration_s", durationS)
            put("top_label", label)
            put("top_display", display)
            put("confidence", confidence)
            put("model_version", modelVersion)
        }
        return writableDatabase.insert("recordings", null, cv)
    }

    fun all(): List<Recording> {
        val out = mutableListOf<Recording>()
        readableDatabase.query(
            "recordings", null, null, null, null, null, "timestamp_ms DESC"
        ).use { c ->
            while (c.moveToNext()) {
                out.add(
                    Recording(
                        id = c.getLong(c.getColumnIndexOrThrow("id")),
                        timestampMs = c.getLong(c.getColumnIndexOrThrow("timestamp_ms")),
                        durationS = c.getInt(c.getColumnIndexOrThrow("duration_s")),
                        topLabel = c.getString(c.getColumnIndexOrThrow("top_label")),
                        topDisplay = c.getString(c.getColumnIndexOrThrow("top_display")),
                        confidence = c.getFloat(c.getColumnIndexOrThrow("confidence")),
                        modelVersion = c.getString(c.getColumnIndexOrThrow("model_version"))
                    )
                )
            }
        }
        return out
    }

    fun delete(id: Long): Int {
        return writableDatabase.delete("recordings", "id=?", arrayOf(id.toString()))
    }
}
