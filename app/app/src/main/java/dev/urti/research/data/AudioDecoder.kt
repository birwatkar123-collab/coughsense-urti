package dev.urti.research.data

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer

/**
 * Decodes an audio file (webm/ogg/m4a/...) via MediaExtractor + MediaCodec into a
 * mono 16 kHz Float32 waveform (samples in [-1, 1]), mirroring the research ffmpeg
 * conversion as closely as the platform allows.
 */
object AudioDecoder {

    class DecodeException(message: String, cause: Throwable? = null) : Exception(message, cause)

    private const val MAX_SAMPLES = AudioConst.SAMPLE_RATE * AudioConst.MAX_SECONDS

    fun decodeToMono16k(context: Context, uri: Uri): FloatArray {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, uri, null)
            var trackIndex = -1
            var mime: String? = null
            for (i in 0 until extractor.trackCount) {
                val fmt = extractor.getTrackFormat(i)
                val m = fmt.getString(MediaFormat.KEY_MIME)
                if (m != null && m.startsWith("audio/")) {
                    trackIndex = i
                    mime = m
                    break
                }
            }
            if (trackIndex < 0 || mime == null) throw DecodeException("no audio track")
            extractor.selectTrack(trackIndex)

            val format = extractor.getTrackFormat(trackIndex)
            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            val pcmEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                format.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } else AudioFormat.ENCODING_PCM_16BIT

            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            try {
                val raw = decodePcmAndRelease(extractor, codec, sampleRate, channelCount, pcmEncoding)
                val mono = downmixAndToFloat(raw, channelCount, pcmEncoding)
                return resample(mono, sampleRate, AudioConst.SAMPLE_RATE)
            } finally {
                codec.stop()
                codec.release()
            }
        } catch (t: DecodeException) {
            throw t
        } catch (t: Exception) {
            throw DecodeException("decode failed: ${t.message}", t)
        } finally {
            extractor.release()
        }
    }

    private fun decodePcmAndRelease(
        extractor: MediaExtractor,
        codec: MediaCodec,
        sampleRate: Int,
        channelCount: Int,
        pcmEncoding: Int,
    ): FloatArray {
        val capacity = sampleRate * channelCount * AudioConst.MAX_SECONDS
        val out = FloatArray(minOf(capacity, MAX_SAMPLES * channelCount))
        var written = 0
        var inputEos = false
        var outputEos = false
        val info = MediaCodec.BufferInfo()

        while (!outputEos) {
            if (!inputEos) {
                val inputIndex = codec.dequeueInputBuffer(10_000L)
                if (inputIndex >= 0) {
                    val buffer = codec.getInputBuffer(inputIndex) ?: continue
                    val size = extractor.readSampleData(buffer, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputEos = true
                    } else {
                        codec.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outputIndex = codec.dequeueOutputBuffer(info, 10_000L)
            when {
                outputIndex >= 0 -> {
                    if (info.size > 0) {
                        val buffer = codec.getOutputBuffer(outputIndex) ?: continue
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        val bps = bytesPerSample(pcmEncoding)
                        val roomSamples = out.size - written
                        val samplesNow = minOf(info.size / bps, roomSamples)
                        if (samplesNow > 0) {
                            convertInto(buffer, pcmEncoding, out, written, samplesNow)
                            written += samplesNow
                        }
                    }
                    codec.releaseOutputBuffer(outputIndex, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputEos = true
                }
                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (inputEos && written == 0) outputEos = true
                }
            }
        }
        return out.copyOf(written)
    }

    private fun bytesPerSample(pcmEncoding: Int): Int =
        if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT) 4 else 2

    private fun convertInto(buffer: ByteBuffer, pcmEncoding: Int, out: FloatArray, offset: Int, sampleCount: Int) {
        var o = offset
        if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT) {
            for (i in 0 until sampleCount) {
                out[o] = coerceFloat(buffer.float)
                o++
            }
        } else {
            for (i in 0 until sampleCount) {
                out[o] = coerceFloat(buffer.short.toFloat() / 32768f)
                o++
            }
        }
    }

    private fun coerceFloat(v: Float): Float = if (v > 1f) 1f else if (v < -1f) -1f else v

    private fun downmixAndToFloat(raw: FloatArray, channelCount: Int, pcmEncoding: Int): FloatArray {
        if (channelCount <= 1) return raw
        val mono = FloatArray(raw.size / channelCount)
        for (i in mono.indices) {
            var s = 0.0
            for (c in 0 until channelCount) s += raw[i * channelCount + c]
            mono[i] = (s / channelCount).toFloat()
        }
        return mono
    }

    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return input
        val outLen = (input.size.toLong() * toRate / fromRate).toInt()
        val ratio = fromRate.toDouble() / toRate.toDouble()
        val result = FloatArray(outLen)
        for (i in 0 until outLen) {
            val pos = i * ratio
            val lo = pos.toInt()
            val hi = minOf(lo + 1, input.size - 1)
            val frac = (pos - lo).toFloat()
            result[i] = input[lo] * (1f - frac) + input[hi] * frac
        }
        return result
    }
}
