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
    val decisionThreshold: Double,
    val band: Band,
    val durationMs: Long,
    val processedSamples: Int,
)

enum class AnalysisSource { RECORDER, IMPORT }

/** Runs the on-device Kaggle multimodel-v2 spectrogram CNN research pipeline. */
object AnalysisEngine {

    fun run(
        samples: FloatArray,
        classifier: KaggleCnnClassifier,
    ): AnalysisOutcome {
        val reason = Preprocessing.rejectReason(samples)
        if (reason != null) return AnalysisOutcome.Rejected(reason, friendly(reason))

        val report = QualityAnalyzer.analyze(samples)
        val prediction = try {
            classifier.predict(samples)
        } catch (_: IllegalArgumentException) {
            return AnalysisOutcome.Rejected(
                "trimmed_too_short",
                "Too little audio remains after removing silence. Please try again.",
            )
        } catch (t: Throwable) {
            return AnalysisOutcome.Rejected(
                "internal_error",
                "Analysis could not be completed (internal error: ${t.javaClass.simpleName}). " +
                    "Please try a shorter recording.",
            )
        }

        return AnalysisOutcome.Completed(
            result = AnalysisResult(
                score = prediction.score,
                positive = prediction.positive,
                decisionThreshold = classifier.metadata.decisionThreshold,
                band = report.band,
                durationMs = report.durationMs,
                processedSamples = prediction.processedSamples,
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
