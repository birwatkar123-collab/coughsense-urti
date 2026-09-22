package dev.urti.research.data

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

data class KaggleModelMetadata(
    val classifierVersion: String,
    val modelFamily: String,
    val embeddingPooling: String,
    val decisionThreshold: Double,
    val calibrated: Boolean,
    val featureCount: Int,
    val researchOnly: Boolean = true,
)

data class KagglePrediction(
    val score: Double,
    val rawScore: Double,
    val positive: Boolean,
    val processedSamples: Int,
)

class KaggleCnnClassifier private constructor(
    private val interpreter: Interpreter,
    private val preprocessor: KaggleSpectrogramPreprocessor,
    private val plattCoef: Double,
    private val plattIntercept: Double,
    val metadata: KaggleModelMetadata,
) : AutoCloseable {

    fun predict(samples: FloatArray): KagglePrediction {
        val spec = preprocessor.transform(samples)
        val input = ByteBuffer.allocateDirect(Float.SIZE_BYTES * spec.size).order(ByteOrder.nativeOrder())
        for (value in spec) input.putFloat(value)
        input.rewind()

        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        val raw = output[0][0].toDouble()
        val calibrated = platt(raw)
        return KagglePrediction(
            score = calibrated,
            rawScore = raw,
            positive = calibrated >= metadata.decisionThreshold,
            processedSamples = preprocessor.targetSamples,
        )
    }

    private fun platt(raw: Double): Double {
        val p = raw.coerceIn(1e-6, 1.0 - 1e-6)
        val logit = ln(p / (1.0 - p))
        val z = plattCoef * logit + plattIntercept
        return 1.0 / (1.0 + exp(-z))
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        fun load(context: Context): KaggleCnnClassifier {
            val cfg = JSONObject(context.assets.open("kaggle_preprocessing.json").bufferedReader().use { it.readText() })
            val cal = JSONObject(context.assets.open("platt_calibration.json").bufferedReader().use { it.readText() })
            val modelBytes = context.assets.open("urti_model.tflite").use { it.readBytes() }
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()
            val interpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(2) })
            val preprocessor = KaggleSpectrogramPreprocessor.fromJson(cfg)
            Log.i(
                TAG,
                "Loaded Kaggle TFLite model. input=${interpreter.getInputTensor(0).shape().contentToString()} " +
                    "output=${interpreter.getOutputTensor(0).shape().contentToString()}",
            )
            val metadata = KaggleModelMetadata(
                classifierVersion = cfg.optString("version", "urti-kaggle-multimodel-v2"),
                modelFamily = "Spectrogram CNN",
                embeddingPooling = "Mel spectrogram",
                decisionThreshold = cfg.optDouble("threshold", 0.5),
                calibrated = cfg.optBoolean("calibrated", true),
                featureCount = preprocessor.nMels * preprocessor.frameCount,
            )
            return KaggleCnnClassifier(
                interpreter = interpreter,
                preprocessor = preprocessor,
                plattCoef = cal.getDouble("coef"),
                plattIntercept = cal.getDouble("intercept"),
                metadata = metadata,
            )
        }

        private const val TAG = "KaggleCnnClassifier"
    }
}

class KaggleSpectrogramPreprocessor private constructor(
    private val sampleRate: Int,
    private val durationSeconds: Int,
    private val nFft: Int,
    private val hopLength: Int,
    val nMels: Int,
    private val fMin: Float,
    private val fMax: Float,
) {
    val targetSamples: Int = sampleRate * durationSeconds
    val frameCount: Int = 1 + targetSamples / hopLength
    private val center = nFft / 2
    private val hann = FloatArray(nFft) { i ->
        (0.5 - 0.5 * cos(2.0 * PI * i.toDouble() / nFft.toDouble())).toFloat()
    }
    private val melFilters = buildMelFilters()

    fun transform(input: FloatArray): FloatArray {
        val trimmed = Preprocessing.trim(input).trimmed
        if (trimmed.size < AudioConst.MIN_SAMPLES) {
            throw IllegalArgumentException("Too little audible content.")
        }

        var maxAbs = 0f
        for (v in trimmed) {
            val a = kotlin.math.abs(v)
            if (a > maxAbs) maxAbs = a
        }
        val normalized = FloatArray(targetSamples)
        val start = max(0, (trimmed.size - targetSamples) / 2)
        val copy = min(targetSamples, trimmed.size - start)
        val denom = max(maxAbs, 1e-8f)
        for (i in 0 until copy) {
            normalized[i] = (trimmed[start + i] / denom).coerceIn(-1f, 1f)
        }

        val melPower = FloatArray(nMels * frameCount)
        var maxPower = 1e-20f
        for (frame in 0 until frameCount) {
            val base = frame * hopLength - center
            val power = powerSpectrum(normalized, base)
            for (mel in 0 until nMels) {
                var acc = 0f
                val filter = melFilters[mel]
                for (bin in filter.indices) {
                    acc += power[bin] * filter[bin]
                }
                val safe = max(acc, 1e-20f)
                melPower[mel * frameCount + frame] = safe
                if (safe > maxPower) maxPower = safe
            }
        }

        val out = FloatArray(nMels * frameCount)
        val refDb = 10f * log10(maxPower)
        for (i in out.indices) {
            val db = (10f * log10(melPower[i]) - refDb).coerceAtLeast(-80f)
            out[i] = (db + 80f) / 80f
        }
        return out
    }

    private fun powerSpectrum(samples: FloatArray, base: Int): FloatArray {
        val bins = nFft / 2 + 1
        val out = FloatArray(bins)
        for (k in 0 until bins) {
            var real = 0.0
            var imag = 0.0
            for (n in 0 until nFft) {
                val idx = base + n
                val sample = if (idx in samples.indices) samples[idx].toDouble() else 0.0
                val windowed = sample * hann[n].toDouble()
                val angle = -2.0 * PI * k.toDouble() * n.toDouble() / nFft.toDouble()
                real += windowed * cos(angle)
                imag += windowed * sin(angle)
            }
            out[k] = (real * real + imag * imag).toFloat()
        }
        return out
    }

    private fun buildMelFilters(): Array<FloatArray> {
        val bins = nFft / 2 + 1
        val weights = Array(nMels) { FloatArray(bins) }
        val fftFreqs = FloatArray(bins) { it.toFloat() * sampleRate.toFloat() / nFft.toFloat() }
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + (melMax - melMin) * i.toFloat() / (nMels + 1).toFloat()
        }
        val hzPoints = FloatArray(melPoints.size) { melToHz(melPoints[it]) }
        for (mel in 0 until nMels) {
            val lower = hzPoints[mel]
            val centerHz = hzPoints[mel + 1]
            val upper = hzPoints[mel + 2]
            val enorm = 2f / (upper - lower)
            for (bin in 0 until bins) {
                val f = fftFreqs[bin]
                val up = (f - lower) / (centerHz - lower)
                val down = (upper - f) / (upper - centerHz)
                weights[mel][bin] = max(0f, min(up, down)) * enorm
            }
        }
        return weights
    }

    private fun hzToMel(hz: Float): Float {
        val fSp = 200f / 3f
        val minLogHz = 1000f
        val minLogMel = minLogHz / fSp
        val logStep = ln(6.4).toFloat() / 27f
        return if (hz < minLogHz) hz / fSp else minLogMel + ln(hz / minLogHz) / logStep
    }

    private fun melToHz(mel: Float): Float {
        val fSp = 200f / 3f
        val minLogHz = 1000f
        val minLogMel = minLogHz / fSp
        val logStep = ln(6.4).toFloat() / 27f
        return if (mel < minLogMel) mel * fSp else minLogHz * exp(logStep * (mel - minLogMel))
    }

    companion object {
        fun fromJson(json: JSONObject): KaggleSpectrogramPreprocessor {
            return KaggleSpectrogramPreprocessor(
                sampleRate = json.getInt("sample_rate"),
                durationSeconds = json.getInt("duration_seconds"),
                nFft = json.getInt("n_fft"),
                hopLength = json.getInt("hop_length"),
                nMels = json.getInt("n_mels"),
                fMin = json.getDouble("fmin").toFloat(),
                fMax = json.getDouble("fmax").toFloat(),
            )
        }
    }
}
