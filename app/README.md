# URTI Research App

Offline-first Android prototype that turns a cough/speech recording into a research-only
YAMNet embedding + V4 linear classifier score. **This is a research prototype, not a medical
device, and never states a diagnosis.**

- Package: `dev.urti.research`
- `minSdk 26` / `targetSdk 36`, Kotlin, Jetpack Compose (Material 3), MVVM
- **No INTERNET permission** — only `RECORD_AUDIO`; all model + weights ship in `assets/`
- On-device pipeline: decode/resample to 16 kHz mono float -> silence/hold trim ->
  quality gate -> YAMNet frame embeddings (dynamic resize) -> mean-pool -> V4 score;
  results stored locally in Room (with delete/clear).

## Model assets (shipped, not downloaded)

| Asset | Size | Notes |
| --- | --- | --- |
| `assets/yamnet_dynamic.tflite` | 12,893,132 B | Official TF-Hub YAMNet converted to dynamic-input TFLite; parity vs TF-Hub verified (max mean-embedding diff 2.9e-6) |
| `assets/v4_weights.json` | 101,258 B | StandardScaler + LogisticRegression (C=0.001, liblinear, balanced) exported from `experiments/v2-source/outputs`; verified max\|diff\| 1.2e-8 vs archived raw scores |

The Kotlin code (trim, reject rules, frame-count formula, mean pooling, sigmoid scoring) was
validated bit-for-bit / numerically against the numpy reference in `../tools` and the archived
research pipeline (see `../APP-STATUS.md`).

## Building

Requires JDK 21 and the Android SDK at `C:\Users\mansi\AppData\Local\Android\Sdk`
(see `local.properties`).

```powershell
$env:JAVA_HOME='C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot'
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
```

Debug APK: `app/build/outputs/apk/debug/app-debug.apk` (~39.7 MB).
Install with `adb install -r app/build/outputs/apk/debug/app-debug.apk`.

## Tests

JVM unit tests verify the on-device pipeline against committed research gold data:

- `PreprocessingTest` — trim start/end and float32 samples **exactly** equal to the numpy
  reference (`src/test/resources/gold/gold_trim.bin`, 23 cases incl. synthetic edge cases and
  16 dev recordings); reject rules; frame-count table.
- `V4ClassifierTest` — loads `gold/v4_fixtures.json` and reproduces archived YAMNet+V4 scores.
- `QualityAnalyzerTest` — quality gate thresholds.

Data used for tests/gold is **train + validation only** (never the test split).