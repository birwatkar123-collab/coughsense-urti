package dev.urti.research.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityAnalyzerTest {

    @Test
    fun silentSignalReportsLowBand() {
        val samples = FloatArray(16000) { 0f }
        val report = QualityAnalyzer.analyze(samples)
        assertEquals(1000L, report.durationMs)
        assertEquals(0f, report.maxAbs, 0f)
        assertEquals(0f, report.rms, 0f)
        assertEquals(Band.LOW, report.band)
    }

    @Test
    fun loudSignalReportsHighBandAndRms() {
        val samples = FloatArray(16000) { 0.5f }
        val report = QualityAnalyzer.analyze(samples)
        assertEquals(Band.HIGH, report.band)
        assertTrue(report.rms == 0.5f)
    }

    @Test
    fun emptyInputDoesNotCrash() {
        val report = QualityAnalyzer.analyze(FloatArray(0))
        assertEquals(0L, report.durationMs)
        assertEquals(Band.LOW, report.band)
    }
}