package dev.urti.research.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import dev.urti.research.ui.AnalysisViewModel
import dev.urti.research.ui.UiState
import java.util.Locale

@Composable
fun RecordScreen(vm: AnalysisViewModel, onBack: () -> Unit, onAnalyzing: () -> Unit) {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    val state by vm.uiState.collectAsState()
    LaunchedEffect(state) {
        if (state != UiState.Idle && state !is UiState.Recording) onAnalyzing()
    }
    LaunchedEffect(Unit) { vm.reset() }

    Scaffold { inner ->
        Column(
            modifier = Modifier.fillMaxSize().padding(inner).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            Text("Record a cough", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Text("Record up to 60 seconds. A clear signal helps the analysis.", style = MaterialTheme.typography.bodyMedium)

            val recording = state as? UiState.Recording
            when {
                recording != null -> {
                    CircularProgressIndicator(modifier = Modifier.padding(top = 16.dp))
                    Text(
                        String.format(Locale.US, "%.1f s", recording.elapsedMs / 1000.0),
                        style = MaterialTheme.typography.headlineMedium,
                    )
                    LinearProgressIndicator(
                        progress = { (recording.level.coerceIn(0f, 1f) * 1f).coerceAtLeast(0.02f) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    OutlinedButton(onClick = { vm.stopAndAnalyze() }) {
                        Text("Stop and analyze")
                    }
                }
                else -> {
                    Box(Modifier.fillMaxWidth().padding(top = 24.dp), contentAlignment = Alignment.Center) {
                        Button(
                            onClick = {
                                if (hasPermission) vm.beginRecording() else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            },
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 32.dp),
                        ) {
                            Text("Start recording", style = MaterialTheme.typography.titleMedium)
                        }
                    }
                    if (!hasPermission) {
                        Text(
                            "Microphone permission is needed to record audio.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }
            OutlinedButton(onClick = onBack) { Text("Back") }
        }
    }
}