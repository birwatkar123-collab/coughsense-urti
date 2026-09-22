package dev.urti.research.ui.screens

import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.urti.research.ui.AnalysisViewModel
import dev.urti.research.ui.UiState
import dev.urti.research.ui.components.QualityCard
import dev.urti.research.ui.components.ScoreCard

@Composable
fun AnalyzeScreen(vm: AnalysisViewModel, onHome: () -> Unit, onBack: () -> Unit) {
    val state by vm.uiState.collectAsState()
    Scaffold { inner ->
        Column(
            modifier = Modifier.fillMaxSize().padding(inner).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            when (val s = state) {
                is UiState.CheckingQuality -> {
                    Heading("Checking audio quality")
                    QualityCard(s.report)
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                is UiState.Preparing -> Phase("Preparing audio", index = 0, totalSteps = 4)
                is UiState.Extracting -> Phase("Extracting mel-spectrogram features", index = 2, totalSteps = 4)
                is UiState.Inferring -> Phase("Running research classifier", index = 3, totalSteps = 4)
                is UiState.Success -> {
                    Heading("Analysis complete")
                    ScoreCard(s.outcome.result)
                    Text(
                        s.outcome.result.positive.takeIf { it }?.let {
                            "The research model output is at or above the reference " +
                                "threshold. This is not a diagnosis."
                        } ?: "The research model output is below the reference threshold.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Button(onClick = onHome, modifier = Modifier.fillMaxWidth()) { Text("Done") }
                    OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Back") }
                }
                is UiState.Failure -> {
                    Heading("Analysis could not be completed")
                    Text(s.message, style = MaterialTheme.typography.bodyLarge)
                    Button(onClick = onHome, modifier = Modifier.fillMaxWidth()) { Text("Back to home") }
                }
                is UiState.Recording, UiState.Idle -> {
                    Heading("Analyzing")
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            }
        }
    }
}

@Composable
private fun Heading(title: String) {
    Text(title, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
}

@Composable
private fun Phase(label: String, index: Int, totalSteps: Int) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        CircularProgressIndicator()
        Text(label, style = MaterialTheme.typography.titleMedium)
        Text(
            "Step $index of $totalSteps",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
