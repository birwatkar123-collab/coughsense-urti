package dev.urti.research.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = Color(0xFF006B5E),
    secondary = Color(0xFF4A635E),
    surface = Color(0xFFFAFAF7),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF59DBC2),
)

@Composable
fun UrtiTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    MaterialTheme(
        colorScheme = if (dark) DarkColors else LightColors,
        content = content,
    )
}