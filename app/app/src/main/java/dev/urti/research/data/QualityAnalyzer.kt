package dev.urti.research.data

import kotlin.math.abs

/** Simple objective audio-quality summary used only for presentation (not diagnosis). */
enum class Band { LOW, MEDIUM, HIGH }

data class QualityReport(
    val durationMs: Long,
    val maxAbs: Float,
    val rms: Float,
    val band: Band,
    val peakFraction: Float,
    val containsSilenceRuns: Boolean,
)

object QualityAnalyzer {

    fun analyze(samples: FloatArray, sampleRate: Int = AudioConst.SAMPLE_RATE): QualityReport {
        val durationMs = samples.size.toLong() * 1000L / sampleRate
        var maxAbs = 0f
        var sumSq = 0.0
        for (v in samples) {
            val a = abs(v)
            if (a > maxAbs) maxAbs = a
            sumSq += v.toDouble() * v.toDouble()
        }
        val rms = if (samples.isEmpty()) 0f else sqrtSafe(sumSq / samples.size)
        val band = when {
            maxAbs < 0.01f -> Band.LOW
            maxAbs < 0.25f -> Band.MEDIUM
            else -> Band.HIGH
        }

        val window = sampleRate / 2
        val step = sampleRate / 20
        var peakIdx = 0
        var peakWin = 0.0
        val silences = ArrayList<Boolean>()
        var i = 0
        while (i < samples.size) {
            val lo = i
            val hi = minOf(i + window, samples.size)
            var wsum = 0.0
            var wmax = 0f
            for (j in lo until hi) {
                val a = abs(samples[j])
                wsum += a
                if (a > wmax) wmax = a
            }
            val winRms = wsum / (hi - lo)
            silences.add(wmax < 0.005f)
            if (winRms > peakWin) {
                peakWin = winRms
                peakIdx = i
            }
            i += step
        }
        val peakFraction = if (samples.isNotEmpty()) peakIdx.toFloat() / samples.size.toFloat() else 0f
        val containsSilenceRuns = if (silences.isNotEmpty()) silences.take(10).any { it } else true

        return QualityReport(
            durationMs = durationMs,
            maxAbs = maxAbs,
            rms = rms,
            band = band,
            peakFraction = peakFraction,
            containsSilenceRuns = containsSilenceRuns,
        )
    }

    private fun sqrtSafe(d: Double): Float {
        return if (d < 0.0) 0f else kotlin.math.sqrt(d).toFloat()
    }
}