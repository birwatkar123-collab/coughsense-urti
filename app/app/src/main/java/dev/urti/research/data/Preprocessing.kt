package dev.urti.research.data

import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Deterministic Float32 mirror of the research V2/V4 CPU preprocessing
 * (see tools/app_trim.py): librosa.effects.trim(y, top_db=30) frame defaults
 * (frame_length=2048, hop_length=512, center=True pad 1024) followed by the
 * accept/reject rules and the 0.96s minimum padding from shared-training-flat.json.
 *
 * Every arithmetic step uses IEEE-754 Float32 in the same order as the numpy
 * reference, so boundaries and trimmed samples are bit-identical.
 */
object AudioConst {
    const val SAMPLE_RATE = 16000
    const val MIN_SAMPLES = 3200        // 0.2 s
    const val PAD_SAMPLES = 15360       // 0.96 s minimum after clipping
    const val SILENT_THRESHOLD = 1e-5f
    const val MAX_SECONDS = 60
}

data class TrimResult(val start: Int, val end: Int, val trimmed: FloatArray)

object Preprocessing {
    const val FRAME_LENGTH = 2048
    const val HOP_LENGTH = 512
    const val CENTER = FRAME_LENGTH / 2
    const val TOP_DB = 30.0f
    private const val AMIN2 = 1e-20f // librosa power_to_db amin = 1e-10, squared

    /** Mirrors tools/app_trim.reject_reasons. Returns a reason or null when accepted. */
    fun rejectReason(samples: FloatArray): String? {
        if (samples.size < AudioConst.MIN_SAMPLES) return "too_short"
        if (samples.size > AudioConst.SAMPLE_RATE * AudioConst.MAX_SECONDS) return "too_long"
        var maxAbs = 0f
        for (v in samples) {
            if (v.isNaN() || v.isInfinite()) return "non_finite"
            val a = abs(v)
            if (a > maxAbs) maxAbs = a
        }
        if (maxAbs < AudioConst.SILENT_THRESHOLD) return "silent"
        return null
    }

    /** Mirrors tools/app_trim.trim_signal (librosa.effects.trim semantics). */
    fun trim(samples: FloatArray): TrimResult {
        val n = samples.size
        if (n == 0) return TrimResult(0, 0, samples)
        val padded = FloatArray(n + 2 * CENTER)
        System.arraycopy(samples, 0, padded, CENTER, n)
        val nFrames = max((padded.size - FRAME_LENGTH) / HOP_LENGTH + 1, 0)
        if (nFrames == 0) return TrimResult(0, n, samples.copyOf())

        val power = FloatArray(nFrames)
        for (f in 0 until nFrames) {
            var acc = 0f
            val base = f * HOP_LENGTH
            for (s in 0 until FRAME_LENGTH) {
                val v = padded[base + s]
                acc += v * v
            }
            power[f] = acc / FRAME_LENGTH.toFloat()
        }

        val rmsSq = FloatArray(nFrames)
        var ref = 0f
        for (i in 0 until nFrames) {
            val r = sqrt(power[i].toDouble()).toFloat()
            val rr = r * r
            rmsSq[i] = rr
            if (rr > ref) ref = rr
        }
        val logMin = (10.0 * log10(max(AMIN2.toDouble(), ref.toDouble()))).toFloat()
        val first = findFirstNonSilent(rmsSq, logMin)
        if (first == -1) return TrimResult(0, n, samples.copyOf())

        val last = findLastNonSilent(rmsSq, logMin)
        val start = first * HOP_LENGTH
        val end = minOf(n, (last + 1) * HOP_LENGTH)
        val out = FloatArray(end - start)
        System.arraycopy(samples, start, out, 0, out.size)
        return TrimResult(start, end, out)
    }

    private fun db(rmsSq: Float, logMin: Float): Float {
        val mx = if (rmsSq > AMIN2) rmsSq.toDouble() else AMIN2.toDouble()
        return 10f * log10(mx).toFloat() - logMin
    }

    private fun findFirstNonSilent(rmsSq: FloatArray, logMin: Float): Int {
        for (i in rmsSq.indices) if (db(rmsSq[i], logMin) > -TOP_DB) return i
        return -1
    }

    private fun findLastNonSilent(rmsSq: FloatArray, logMin: Float): Int {
        for (i in rmsSq.indices.reversed()) if (db(rmsSq[i], logMin) > -TOP_DB) return i
        return -1
    }

    /** Mirrors tools/app_trim.preprocess_to_embedding_input (trim -> clip -> min pad). */
    data class Preprocessed(val ok: Boolean, val reason: String?, val samples: FloatArray) {
        val isEmpty: Boolean get() = samples.isEmpty()
    }

    fun preprocess(samples: FloatArray): Preprocessed {
        val reason = rejectReason(samples)
        if (reason != null) return Preprocessed(false, reason, FloatArray(0))
        val (_, _, trimmed) = trim(samples)
        if (trimmed.size < AudioConst.MIN_SAMPLES) {
            return Preprocessed(false, "trimmed_too_short", FloatArray(0))
        }
        for (i in trimmed.indices) {
            if (trimmed[i] > 1f) trimmed[i] = 1f else if (trimmed[i] < -1f) trimmed[i] = -1f
        }
        val padded = if (trimmed.size < AudioConst.PAD_SAMPLES) {
            FloatArray(AudioConst.PAD_SAMPLES)
        } else trimmed
        if (padded.size == AudioConst.PAD_SAMPLES && trimmed.size < AudioConst.PAD_SAMPLES) {
            System.arraycopy(trimmed, 0, padded, 0, trimmed.size)
        }
        return Preprocessed(true, null, padded)
    }

    /** YAMNet window frame count for a waveform length; used for output buffer sizing. */
    fun yamnetFrameCount(length: Int): Int {
        return if (length < 15600) 1 else 1 + (length - 15600 + 7679) / 7680
    }
}