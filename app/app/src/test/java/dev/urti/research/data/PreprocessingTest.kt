package dev.urti.research.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Validates the Kotlin trim/preprocess mirror bit-for-bit against the numpy reference
 * (tools/app_trim.py), which itself was validated end-to-end against the archived
 * research embeddings (tools/parity_check.py, PARITY PASS).
 */
class PreprocessingTest {

    @Test
    fun trimMatchesGoldReferenceExactly() {
        val cases = GoldTrimReader.read("/gold/gold_trim.bin")
        assertTrue("gold file should contain cases", cases.isNotEmpty())

        for (c in cases) {
            val got = Preprocessing.trim(c.input)
            val msg = "${c.id}: expected start=${c.start} end=${c.end} got start=${got.start} end=${got.end}"
            assertEquals(msg, c.start, got.start)
            assertEquals(msg, c.end, got.end)
            assertArrayEquals("${c.id}: trimmed float32 samples differ", c.out, got.trimmed, 0f)
        }
    }

    @Test
    fun rejectRulesMatchResearch() {
        assertEquals("too_short", Preprocessing.rejectReason(FloatArray(100)))
        assertEquals("silent", Preprocessing.rejectReason(FloatArray(AudioConst.MIN_SAMPLES)))
        assertEquals("too_long", Preprocessing.rejectReason(FloatArray(AudioConst.SAMPLE_RATE * 61)))
        assertEquals(null, Preprocessing.rejectReason(FloatArray(AudioConst.MIN_SAMPLES) { 0.2f }))
        val withNan = FloatArray(AudioConst.MIN_SAMPLES) { 0.2f }
        withNan[5] = Float.NaN
        assertEquals("non_finite", Preprocessing.rejectReason(withNan))
    }

    @Test
    fun preprocessPadsShortValidSignalsToMin() {
        val samples = FloatArray(5000) { 0.2f }
        val pre = Preprocessing.preprocess(samples)
        assertTrue(pre.ok)
        assertEquals(AudioConst.PAD_SAMPLES, pre.samples.size)
    }

    @Test
    fun preprocessRejectsSilent() {
        val pre = Preprocessing.preprocess(FloatArray(16000) { 0f })
        assertTrue(!pre.ok)
        assertEquals("silent", pre.reason)
    }

    @Test
    fun yamnetWindowFrameCount() {
        val expected = mapOf(
            15360 to 1, 16000 to 2, 32000 to 4, 48000 to 6,
            100000 to 12, 480000 to 62, 960000 to 124,
        )
        for ((len, frames) in expected) {
            assertEquals("frame count for $len", frames, Preprocessing.yamnetFrameCount(len))
        }
    }
}

class GoldTrimReader private constructor(private val buf: ByteBuffer) {
    companion object {
        fun read(resource: String): List<GoldCase> {
            val raw = resourceBytes(resource)
            val buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            check(buf.remaining() >= 8) { "too few bytes (${buf.remaining()}) for magic" }
            val magic = ByteArray(8)
            buf.get(magic)
            if (String(magic, Charsets.US_ASCII) != "URTIGOLD") {
                val hex = magic.joinToString(",") { (it.toInt() and 0xFF).toString() }
                error("bad magic read='${String(magic, Charsets.US_ASCII)}' hex=[$hex] total=${raw.size} " +
                        "classpath=" + System.getProperty("java.class.path"))
            }
            val n = buf.int
            val items = ArrayList<GoldCase>(n)
            for (i in 0 until n) {
                val idLen = buf.int
                val id = readAscii(buf, idLen)
                val inLen = buf.int
                val input = FloatArray(inLen) { buf.float }
                val outLen = buf.int
                val out = FloatArray(outLen) { buf.float }
                val start = buf.int
                val end = buf.int
                items.add(GoldCase(id, input, out, start, end))
            }
            return items
        }

        private fun resourceBytes(resource: String): ByteArray {
            val viaClasspath = GoldTrimReader::class.java.getResourceAsStream(resource)?.readBytes()
            if (viaClasspath != null) {
                System.err.println("[gold] read from classpath, ${viaClasspath.size} bytes, head=" +
                        String(viaClasspath.copyOfRange(0, 8), kotlin.text.Charsets.US_ASCII))
                return viaClasspath
            }
            // AGP may exclude .bin from merged resources; unit tests run with cwd = module dir.
            val file = java.io.File("src/test/resources", resource.trimStart('/'))
            if (file.exists()) {
                System.err.println("[gold] read from file ${file.absolutePath}, ${file.length()} bytes")
                return file.readBytes()
            }
            val cwd = System.getProperty("user.dir")
            error("resource not found: $resource (cwd=$cwd, file=$file exists=${file.exists()})")
        }

        private fun readAscii(buf: ByteBuffer, len: Int): String {
            val bytes = ByteArray(len) { buf.get() }
            return String(bytes, Charsets.US_ASCII)
        }
    }
}

data class GoldCase(
    val id: String,
    val input: FloatArray,
    val out: FloatArray,
    val start: Int,
    val end: Int,
)