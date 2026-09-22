package dev.urti.research.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.urti.research.data.AnalysisResult
import dev.urti.research.data.Band
import dev.urti.research.data.QualityReport
import java.util.Locale

fun formatSeconds(ms: Long): String = String.format(Locale.US, "%.1f s", ms / 1000.0)

fun levelLabel(maxAbs: Float): String = when {
    maxAbs < 0.01f -> "Very low"
    maxAbs < 0.1f -> "Low"
    maxAbs < 0.3f -> "Medium"
    else -> "High"
}

fun bandLabel(band: Band): String = when (band) {
    Band.LOW -> "Low"
    Band.MEDIUM -> "Medium"
    Band.HIGH -> "High"
}

@Composable
fun ResearchDisclaimer(modifier: Modifier = Modifier) {
    Card(modifier = modifier.fillMaxWidth(), colors = CardDefaults.cardColors()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Research prototype", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(
                "This app is an experimental cough-analysis prototype. The output is a research " +
                    "model score and is not a medical diagnosis. Do not use it to make health decisions.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
fun QualityCard(report: QualityReport, modifier: Modifier = Modifier) {
    Card(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Audio quality", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Duration")
                Text(formatSeconds(report.durationMs))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Signal level")
                Text(levelLabel(report.maxAbs))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Amplitude band")
                Text(bandLabel(report.band))
            }
            Text(
                "Quality checks are used for display only and do not determine the model score.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
fun ScoreCard(result: AnalysisResult, modifier: Modifier = Modifier) {
    Card(modifier = modifier.fillMaxWidth(), colors = CardDefaults.cardColors()) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Research model score", style = MaterialTheme.typography.titleMedium)
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    text = String.format(Locale.US, "%.3f", result.score),
                    style = MaterialTheme.typography.displayMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text("  / 1.0", style = MaterialTheme.typography.bodyMedium)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Amplitude band")
                Text(bandLabel(result.band))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Audio duration")
                Text(formatSeconds(result.durationMs))
            }
            Text(
                "Reference decision threshold: 0.5. This is an experimental output of the " +
                    "research classifier, not a clinical finding.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}