package dev.urti.research.data

sealed interface AnalysisOutcome {
    data class Completed(
        val result: AnalysisResult,
        val report: QualityReport,
    ) : AnalysisOutcome

    data class Rejected(val reason: String, val message: String) : AnalysisOutcome
}

data class AnalysisResult(
    val score: Double,
    val positive: Boolean,
    val band: Band,
    val durationMs: Long,
    val processedSamples: Int,
)

enum class AnalysisSource { RECORDER, IMPORT }

/** Runs the on-device research pipeline exactly as validated in tools/parity_check.py. */
object AnalysisEngine {

    fun run(
        samples: FloatArray,
        embedder: YamnetEmbedder,
        classifier: V4Classifier,
    ): AnalysisOutcome {
        val reason = Preprocessing.rejectReason(samples)
        if (reason != null) return AnalysisOutcome.Rejected(reason, friendly(reason))

        val report = QualityAnalyzer.analyze(samples)

        val prepped = Preprocessing.preprocess(samples)
        if (!prepped.ok) {
            val r = prepped.reason ?: "preprocess_failed"
            if (r == "trimmed_too_short") {
                return AnalysisOutcome.Rejected(r, "Too little audio remains after removing silence. Please try again.")
            }
            return AnalysisOutcome.Rejected(r, friendly(r))
        }

        val embedding = try {
            embedder.meanEmbedding(prepped.samples)
        } catch (t: Throwable) {
            return AnalysisOutcome.Rejected(
                "internal_error",
                "Analysis could not be completed (internal error: ${t.javaClass.simpleName}). " +
                    "Please try a shorter recording.",
            )
        }
        val score = classifier.score(embedding)
        val positive = classifier.isPositive(score)

        return AnalysisOutcome.Completed(
            result = AnalysisResult(
                score = score,
                positive = positive,
                band = report.band,
                durationMs = report.durationMs,
                processedSamples = prepped.samples.size,
            ),
            report = report,
        )
    }

    private fun friendly(reason: String): String = when (reason) {
        "too_short" -> "Recording is too short (at least 0.2 seconds is needed)."
        "too_long" -> "Recording is longer than 60 seconds."
        "silent" -> "No audible audio was detected. Please try recording closer to the microphone."
        "non_finite" -> "Audio could not be read."
        else -> "Analysis could not be completed."
    }
}