package dev.urti.research.ui

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.urti.research.UrtiApp
import dev.urti.research.data.AnalysisEngine
import dev.urti.research.data.AnalysisEntity
import dev.urti.research.data.AnalysisOutcome
import dev.urti.research.data.AnalysisSource
import dev.urti.research.data.AudioDecoder
import dev.urti.research.data.AudioRecorder
import dev.urti.research.data.KaggleCnnClassifier
import dev.urti.research.data.QualityReport
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface UiState {
    data object Idle : UiState
    data class Recording(val elapsedMs: Long, val level: Float) : UiState
    data object Preparing : UiState
    data class CheckingQuality(val report: QualityReport) : UiState
    data object Extracting : UiState
    data object Inferring : UiState
    data class Success(val report: QualityReport, val outcome: AnalysisOutcome.Completed, val source: AnalysisSource) : UiState
    data class Failure(val message: String) : UiState
}

class AnalysisViewModel(app: Application) : AndroidViewModel(app) {

    private val recorder = AudioRecorder()
    private val classifier: KaggleCnnClassifier by lazy { UrtiApp.instance.classifier }
    private val historyDao = UrtiApp.instance.database.historyDao()

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    val history = historyDao.observeAll()

    fun classifierMetadata() = classifier.metadata

    fun deleteHistory(entity: dev.urti.research.data.AnalysisEntity) {
        viewModelScope.launch { historyDao.delete(entity) }
    }

    fun clearHistory() {
        viewModelScope.launch { historyDao.clear() }
    }

    private var tickerJob: kotlinx.coroutines.Job? = null

    fun reset() {
        tickerJob?.cancel()
        _uiState.value = UiState.Idle
    }

    fun beginRecording() {
        if (recorder.isRecording || _uiState.value !is UiState.Idle) return
        val ok = recorder.start()
        if (!ok) {
            _uiState.value = UiState.Failure("Could not start the microphone.")
            return
        }
        val started = System.currentTimeMillis()
        _uiState.value = UiState.Recording(0L, 0f)
        tickerJob = viewModelScope.launch {
            while (recorder.isRecording) {
                delay(100)
                val st = _uiState.value
                if (st is UiState.Recording) {
                    _uiState.value = st.copy(elapsedMs = System.currentTimeMillis() - started, level = recorder.level)
                }
            }
        }
    }

    fun stopAndAnalyze() {
        tickerJob?.cancel()
        if (!recorder.isRecording) return
        val recording = recorder.stopAndCollect()
        analyze(recording.samples, AnalysisSource.RECORDER)
    }

    fun analyzeImported(uri: Uri) {
        val app = getApplication<Application>()
        _uiState.value = UiState.Preparing
        viewModelScope.launch {
            val samples = try {
                withContext(Dispatchers.IO) { AudioDecoder.decodeToMono16k(app, uri) }
            } catch (t: Throwable) {
                _uiState.value = UiState.Failure("Analysis could not be completed.")
                return@launch
            }
            analyze(samples, AnalysisSource.IMPORT)
        }
    }

    private fun analyze(samples: FloatArray, source: AnalysisSource) {
        if (_uiState.value !is UiState.Preparing) _uiState.value = UiState.Preparing
        viewModelScope.launch {
            val report = withContext(Dispatchers.Default) {
                dev.urti.research.data.QualityAnalyzer.analyze(samples)
            }
            _uiState.value = UiState.CheckingQuality(report)
            delay(1100)
            _uiState.value = UiState.Extracting
            delay(700)
            _uiState.value = UiState.Inferring
            val outcome = try {
                withContext(Dispatchers.Default) {
                    AnalysisEngine.run(samples, classifier)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Analysis failed", t)
                _uiState.value = UiState.Failure("Analysis could not be completed (${t.javaClass.simpleName}).")
                return@launch
            }
            when (outcome) {
                is AnalysisOutcome.Completed -> {
                    val id = historyDao.insert(
                        AnalysisEntity(
                            timestampMs = System.currentTimeMillis(),
                            source = if (source == AnalysisSource.RECORDER) "recorder" else "import",
                            durationMs = outcome.result.durationMs,
                            score = outcome.result.score,
                            positive = outcome.result.positive,
                            band = outcome.result.band.name,
                        )
                    )
                    _uiState.value = UiState.Success(
                        report = outcome.report,
                        outcome = outcome,
                        source = source,
                    )
                }
                is AnalysisOutcome.Rejected -> {
                    _uiState.value = UiState.Failure(outcome.message)
                }
            }
        }
    }

    companion object {
        private const val TAG = "AnalysisViewModel"
    }
}
