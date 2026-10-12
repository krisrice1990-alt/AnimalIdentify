package com.animalidentify.app

import android.app.Activity
import android.app.AlertDialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class HistoryActivity : Activity() {

    private lateinit var listView: ListView
    private lateinit var emptyText: TextView
    private var db: DetectionDb? = null
    private val bg = Executors.newSingleThreadExecutor()
    private val records = mutableListOf<Recording>()
    private lateinit var adapter: ArrayAdapter<Recording>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        listView = findViewById(R.id.historyList)
        emptyText = findViewById(R.id.emptyText)
        db = DetectionDb(this)

        adapter = object : ArrayAdapter<Recording>(
            this, android.R.layout.simple_list_item_2, android.R.id.text1, records
        ) {
            private val fmt =
                SimpleDateFormat("MMM d, HH:mm", Locale.getDefault())

            override fun getView(
                position: Int, convertView: View?, parent: ViewGroup
            ): View {
                val v = convertView ?: LayoutInflater.from(context)
                    .inflate(android.R.layout.simple_list_item_2, parent, false)
                val r = getItem(position)!!
                v.findViewById<TextView>(android.R.id.text1).apply {
                    text = "${r.topDisplay} — ${(r.confidence * 100).toInt()}%"
                    setTextColor(0xFFF0F0F0.toInt())
                }
                v.findViewById<TextView>(android.R.id.text2).apply {
                    text = fmt.format(Date(r.timestampMs))
                    setTextColor(0xFFA0A0A0.toInt())
                }
                return v
            }
        }
        listView.adapter = adapter

        listView.setOnItemLongClickListener { _, _, pos, _ ->
            confirmDelete(records[pos])
            true
        }

        load()
    }

    private fun load() {
        bg.execute {
            val all = try {
                db?.all() ?: emptyList()
            } catch (e: Exception) {
                emptyList()
            }
            runOnUiThread {
                records.clear()
                records.addAll(all)
                adapter.notifyDataSetChanged()
                emptyText.visibility =
                    if (records.isEmpty()) View.VISIBLE else View.GONE
            }
        }
    }

    private fun confirmDelete(r: Recording) {
        AlertDialog.Builder(this)
            .setMessage(getString(R.string.delete_confirm))
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                bg.execute {
                    try { db?.delete(r.id) } catch (_: Exception) {}
                    load()
                }
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.execute { try { db?.close() } catch (_: Exception) {} }
        bg.shutdown()
    }
}
