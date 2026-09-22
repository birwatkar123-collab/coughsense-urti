package dev.urti.research.data

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.math.abs

/** Records mono 16 kHz PCM16 from the microphone into Float32 samples. */
class AudioRecorder {

    data class Recording(val samples: FloatArray, val durationMs: Long)

    private var record: AudioRecord? = null
    private var thread: Thread? = null

    @Volatile
    private var running = false

    @Volatile
    var level: Float = 0f
        private set

    private val buffer = ArrayList<Float>()

    val isRecording: Boolean get() = running

    fun start(): Boolean {
        val minBuf = AudioRecord.getMinBufferSize(
            AudioConst.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            AudioConst.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf * 2, 4096),
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) return false
        record = rec
        buffer.clear()
        level = 0f
        rec.startRecording()
        running = true
        thread = Thread {
            val shorts = ShortArray(2048)
            val floats = FloatArray(2048)
            var maxAbs = 0f
            while (running) {
                val read = rec.read(shorts, 0, shorts.size)
                if (read <= 0) continue
                for (i in 0 until read) {
                    val v = shorts[i].toFloat() / 32768f
                    floats[i] = v
                    val a = abs(v)
                    if (a > maxAbs) maxAbs = a
                }
                synchronized(buffer) {
                    for (i in 0 until read) buffer.add(floats[i])
                    if (buffer.size >= AudioConst.SAMPLE_RATE * AudioConst.MAX_SECONDS) {
                        running = false
                    }
                }
                level = maxAbs
                maxAbs = 0f
            }
            try {
                rec.stop()
            } catch (_: Exception) {
            }
            rec.release()
        }.also {
            it.isDaemon = true
            it.start()
        }
        return true
    }

    fun stopAndCollect(): Recording {
        running = false
        try {
            thread?.join(2000)
        } catch (_: InterruptedException) {
        }
        val samples: FloatArray
        val durationMs: Long
        synchronized(buffer) {
            samples = FloatArray(buffer.size) { buffer[it] }
            durationMs = samples.size.toLong() * 1000L / AudioConst.SAMPLE_RATE
            buffer.clear()
        }
        return Recording(samples, durationMs)
    }
}