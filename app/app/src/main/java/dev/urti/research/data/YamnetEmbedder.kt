package dev.urti.research.data

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-device YAMNet (official tfhub.google.com/yamnet/1) converted to a dynamic-length
 * TFLite model. Mirrors tools/parity_check.py: input is the processed mono-16k waveform,
 * output 0 is frame embeddings [frames, 1024]. Mean pooling is applied explicitly in
 * Float32 so it matches the research pipeline bit-for-bit.
 */
class YamnetEmbedder private constructor(
    private val interpreter: Interpreter,
    private val inputIndex: Int,
    private val embeddingOutputIndex: Int?,
    private val framesOutputIndex: Int,
) : AutoCloseable {

    companion object {
        private const val EMBED_DIM = 1024
        private const val TAG = "YamnetEmbedder"

        @android.annotation.SuppressLint("DiscouragedApi")
        fun load(context: Context): YamnetEmbedder {
            val bytes = context.assets.open("yamnet_dynamic.tflite").use { it.readBytes() }
            val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            buffer.put(bytes)
            buffer.rewind()

            val options = Interpreter.Options().apply {
                setUseNNAPI(false)
                try {
                    setUseXNNPACK(false)
                } catch (e: NoSuchMethodError) {
                    // Older TFLite versions may not have this method
                }
                setNumThreads(1)
            }
            val interpreter = Interpreter(buffer, options)

            val inDetails = interpreter.getInputTensor(0)
            Log.d(TAG, "Input tensor shape: ${inDetails.shape().joinToString(",")}, type: ${inDetails.dataType()}")

            val outCount = interpreter.getOutputTensorCount()
            for (i in 0 until outCount) {
                val t = interpreter.getOutputTensor(i)
                Log.d(TAG, "Output $i: shape=${t.shape().joinToString(",")}, type=${t.dataType()}")
            }

            val embeddingOutputIndex = (0 until outCount).firstOrNull { i ->
                val shape = interpreter.getOutputTensor(i).shape()
                shape.size == 1 && shape[0] == EMBED_DIM
            }
            val framesOutputIndex = (0 until outCount).firstOrNull { i ->
                val shape = interpreter.getOutputTensor(i).shape()
                shape.size == 2 && shape[1] == EMBED_DIM
            } ?: 0

            return YamnetEmbedder(interpreter, inputIndex = 0, embeddingOutputIndex, framesOutputIndex)
        }
    }

    fun frameCount(length: Int): Int = Preprocessing.yamnetFrameCount(length)

    /** Runs the model and returns the mean-pooled 1024-d embedding (Float32 mean over frames). */
    fun meanEmbedding(waveform: FloatArray): FloatArray {
        val frames = frameCount(waveform.size)
        // The exported YAMNet model accepts a dynamic 1D waveform. Keeping the
        // input rank at 1 is required by the model's internal PAD operation.
        interpreter.resizeInput(inputIndex, intArrayOf(waveform.size))
        interpreter.allocateTensors()

        val inShape = interpreter.getInputTensor(inputIndex).shape()
        Log.d(TAG, "Resized input shape: ${inShape.joinToString(",")}")

        embeddingOutputIndex?.let { outputIndex ->
            val emb = FloatArray(EMBED_DIM)
            try {
                interpreter.runForMultipleInputsOutputs(arrayOf(waveform), mapOf(outputIndex to emb))
            } catch (t: Throwable) {
                Log.e(TAG, "interpreter.run failed: ${t.message}", t)
                throw t
            }
            return emb
        }

        val outFrames = Array(frames) { FloatArray(EMBED_DIM) }
        val outputs = mapOf(framesOutputIndex to outFrames)
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(waveform), outputs)
        } catch (t: Throwable) {
            Log.e(TAG, "interpreter.run failed: ${t.message}", t)
            throw t
        }

        val emb = FloatArray(EMBED_DIM)
        for (d in 0 until EMBED_DIM) {
            var acc = 0f
            for (f in 0 until frames) acc += outFrames[f][d]
            emb[d] = acc / frames.toFloat()
        }
        return emb
    }

    override fun close() {
        interpreter.close()
    }
}
