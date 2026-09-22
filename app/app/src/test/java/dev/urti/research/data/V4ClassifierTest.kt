package dev.urti.research.data

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Checks the App port of the V4 classifier reproduces the archived research raw scores
 * (fixtures were generated from the archived dev/validation predictions).
 */
class V4ClassifierTest {

    @Test
    fun scoreMatchesArchivedRawScores() {
        val classifier = loadClassifier("/model/v4_weights.json")
        val fixtures = JSONObject(resourceText("/gold/v4_fixtures.json"))

        assertTrue(classifier.metadata.researchOnly)
        assertEquals("YAMNet", classifier.metadata.modelFamily)
        assertEquals("mean", classifier.metadata.embeddingPooling)
        assertEquals("V4", classifier.metadata.classifierVersion)
        assertEquals(classifier.metadata.featureCount, classifier.metadata.featureCount)
        assertEquals(1024, classifier.metadata.featureCount)

        val arr = fixtures.getJSONArray("fixtures")
        assertTrue("fixtures must be present", arr.length() > 0)
        for (i in 0 until arr.length()) {
            val f = arr.getJSONObject(i)
            val embedding = doublesToFloats(f.getJSONArray("embedding"))
            val expected = f.getDouble("expected_raw_score")
            val got = classifier.score(embedding)
            assertEquals("fixture ${f.getString("uuid")}", expected, got, 1e-6)
        }
    }

    private fun loadClassifier(path: String): V4Classifier {
        return V4Classifier.load(JSONObject(resourceText(path)))
    }

    private fun resourceText(path: String): String {
        return V4ClassifierTest::class.java.getResourceAsStream(path)?.bufferedReader()?.use { it.readText() }
            ?: error("resource not found: $path")
    }

    private fun doublesToFloats(arr: JSONArray): FloatArray {
        val out = FloatArray(arr.length())
        for (i in 0 until arr.length()) out[i] = arr.getDouble(i).toFloat()
        return out
    }
}