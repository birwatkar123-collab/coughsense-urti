package dev.urti.research.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.urti.research.data.AnalysisEntity
import dev.urti.research.ui.AnalysisViewModel
import dev.urti.research.ui.components.formatSeconds
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryScreen(vm: AnalysisViewModel, onBack: () -> Unit) {
    val entries by vm.history.collectAsState(initial = emptyList())
    Scaffold { inner ->
        Column(
            modifier = Modifier.fillMaxSize().padding(inner).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text("Analysis history", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                TextButton(
                    onClick = { vm.clearHistory() },
                    enabled = entries.isNotEmpty(),
                ) { Text("Clear all") }
            }
            Text(
                "History is stored only on this device.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                items(entries, key = { it.id }) { e ->
                    HistoryItem(e, onDelete = { vm.deleteHistory(e) })
                }
                if (entries.isEmpty()) {
                    item {
                        Text("No analyses yet.", style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
            OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Back") }
        }
    }
}

@Composable
private fun HistoryItem(entity: AnalysisEntity, onDelete: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(
                    SimpleDateFormat("MMM dd, HH:mm", Locale.US).format(Date(entity.timestampMs)),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    String.format(Locale.US, "score %.3f", entity.score),
                    style = MaterialTheme.typography.titleSmall,
                )
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Source: ${entity.source}", style = MaterialTheme.typography.bodySmall)
                Text("Duration: ${formatSeconds(entity.durationMs)}", style = MaterialTheme.typography.bodySmall)
                Text("Band: ${entity.band}", style = MaterialTheme.typography.bodySmall)
            }
            TextButton(onClick = onDelete) { Text("Delete") }
        }
    }
}