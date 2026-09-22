package dev.urti.research.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.urti.research.ui.components.ResearchDisclaimer

@Composable
fun HomeScreen(onNavigate: (String) -> Unit) {
    Scaffold { inner ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner)
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(Modifier.fillMaxWidth()) {}
            Text("URTI Research", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text(
                "Analyze cough sounds on-device with the research model. " +
                    "Everything stays on this phone.",
                style = MaterialTheme.typography.bodyLarge,
            )
            ResearchDisclaimer()
            Button(
                onClick = { onNavigate("record") },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Record a cough") }
            OutlinedButton(
                onClick = { onNavigate("import") },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Import an audio file") }
            OutlinedButton(
                onClick = { onNavigate("history") },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Analysis history") }
            OutlinedButton(
                onClick = { onNavigate("about") },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("About this model") }
        }
    }
}