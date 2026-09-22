package dev.urti.research.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.urti.research.data.KaggleModelMetadata
import dev.urti.research.ui.components.ResearchDisclaimer

@Composable
fun AboutScreen(metadata: KaggleModelMetadata, onBack: () -> Unit) {
    Scaffold { inner ->
        Column(
            modifier = Modifier.fillMaxSize().padding(inner).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("About the model", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            ResearchDisclaimer()
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Feature extractor")
                Text(metadata.modelFamily)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Input features")
                Text(metadata.embeddingPooling)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Classifier version")
                Text(metadata.classifierVersion)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Feature count")
                Text("${metadata.featureCount}")
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Reference threshold")
                Text(String.format(java.util.Locale.US, "%.2f", metadata.decisionThreshold))
            }
            Text(
                "All model weights and the feature extractor run entirely on this device. " +
                    "No audio is uploaded. Research validity: prototype only.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Back") }
        }
    }
}
