package dev.urti.research.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.urti.research.ui.AnalysisViewModel
import dev.urti.research.ui.UiState

@Composable
fun ImportScreen(vm: AnalysisViewModel, onBack: () -> Unit, onAnalyzing: () -> Unit) {
    val state by vm.uiState.collectAsState()
    LaunchedEffect(state) {
        if (state != UiState.Idle) onAnalyzing()
    }
    LaunchedEffect(Unit) { vm.reset() }

    val openLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            vm.analyzeImported(uri)
        }
    }

    Scaffold { inner ->
        Column(
            modifier = Modifier.fillMaxSize().padding(inner).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Import an audio file", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Text(
                "Pick a cough recording from your device (wav, m4a, ogg, webm, ...). " +
                    "The file is decoded and analyzed fully offline.",
                style = MaterialTheme.typography.bodyMedium,
            )
            Button(
                onClick = { openLauncher.launch(arrayOf("audio/*")) },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Choose a file") }
            OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Back") }
        }
    }
}