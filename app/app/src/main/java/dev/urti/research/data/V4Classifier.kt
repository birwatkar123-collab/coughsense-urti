package dev.urti.research.data

import org.json.JSONObject

data class ModelMetadata(
    val modelFamily: String,
    val embeddingPooling: String,
    val classifierVersion: String,
    val researchOnly: Boolean,
    val featureCount: Int,
    val positiveClass: Int,
    val decisionThreshold: Double,
)

/** V4 raw classifier: StandardScaler (mean/scale) + LogisticRegression weights. */
class V4Classifier(
    private val coef: DoubleArray,
    private val intercept: Double,
    private val mean: DoubleArray,
    private val scale: DoubleArray,
    val metadata: ModelMetadata,
) {

    /** sigmoid((embedding - mean)/scale . coef + intercept) in Double, mirroring research V4. */
    fun score(embedding: FloatArray): Double {
        var z = intercept
        for (i in coef.indices) {
            val x = (embedding[i].toDouble() - mean[i]) / scale[i]
            z += x * coef[i]
        }
        return 1.0 / (1.0 + kotlin.math.exp(-z))
    }

    fun isPositive(score: Double): Boolean = score >= metadata.decisionThreshold

    companion object {
        private const val FEATURE_COUNT = 1024

        fun load(json: JSONObject): V4Classifier {
            val logistic = json.getJSONObject("logistic")
            val scaler = json.getJSONObject("scaler")
            val coef = toDoubleArray(logistic.getJSONArray("coef"))
            val intercept = logistic.getJSONArray("intercept").getDouble(0)
            val mean = toDoubleArray(scaler.getJSONArray("mean"))
            val scale = toDoubleArray(scaler.getJSONArray("scale"))
            require(coef.size == FEATURE_COUNT && mean.size == FEATURE_COUNT && scale.size == FEATURE_COUNT) {
                "unexpected V4 feature count ${coef.size}/${mean.size}/${scale.size}"
            }
            val meta = ModelMetadata(
                modelFamily = json.optString("model_family", "YAMNet"),
                embeddingPooling = json.optString("embedding_pooling", "mean"),
                classifierVersion = json.optString("classifier_version", "V4"),
                researchOnly = json.optBoolean("research_only", true),
                featureCount = json.optInt("feature_count", FEATURE_COUNT),
                positiveClass = json.optInt("positive_class", 1),
                decisionThreshold = json.optDouble("decision_threshold", 0.5),
            )
            return V4Classifier(coef, intercept, mean, scale, meta)
        }

        @android.annotation.SuppressLint("DiscouragedApi")
        fun load(context: android.content.Context): V4Classifier {
            val text = context.assets.open("v4_weights.json").bufferedReader().use { it.readText() }
            return load(JSONObject(text))
        }

        private fun toDoubleArray(arr: org.json.JSONArray): DoubleArray {
            val out = DoubleArray(arr.length())
            for (i in 0 until arr.length()) out[i] = arr.getDouble(i)
            return out
        }
    }
}