# APP-STATUS — URTI Research Prototype (Android)

Status date: 2026-09-21. This is the final milestone report for the Android research app
deliverable. It covers what shipped, how it was verified, and the constraint checklist.

## Delivered

| Item | Result |
| --- | --- |
| Android project `D:\urti\app` (Gradle 8.14.3 wrapper, AGP 8.11.1, Kotlin 2.1.0/KSP, Compose BOM 2025.05.01, Room 2.6.1, TFLite 2.14.0) | Builds clean on JDK 21 |
| Debug APK | `app/build/outputs/apk/debug/app-debug.apk` — 39,739,462 B (~39.7 MB), `versionName 0.1.0` |
| JVM unit tests | 9/9 passing against committed gold/fixtures |
| Assets embedded in APK | `assets/yamnet_dynamic.tflite`, `assets/v4_weights.json` |
| Features | Record / import (API 26+ long-form or uncompressed formats) / analyze / local Room history w/ delete+clear / about screen |

## Pipeline fidelity (the critical part)

The on-device code is a faithful mirror of the archived research reference
(`experiments/v2-source`), not a re-derivation:

1. 16 kHz mono float (MediaCodec decode or AudioRecord PCM16).
2. Silence trim `top_db=30`, frame 2048, hop 512, center-pad 1024 — implemented in **sequential
   Float32** identical to the numpy port `tools/app_trim.py` (librosa cannot be imported on this
   Windows machine — Windows AppControl policy blocks `sklearn.cluster` DLLs, which librosa needs;
   the numpy port is the validated fallback).
3. Reject rules: too_short (<3200), silent (`max|y|<1e-5`), non-finite; too_long (>60 s).
4. Pad to >=15360 samples; YAMNet dynamic `[frames,1024]`; mean-pool; V4 sigmoid score.

Verification numbers (all train/validation only):

| Check | Tool | Result |
| --- | --- | --- |
| Trim port self-test | `tools/_selftest_trim.py` | PASS (burst 3.2s/8.2s edges, silent/too_short/too_long rules, 15360 pad) |
| End-to-end parity vs archived pipeline | `tools/parity_check.py` (12 dev rows) | PASS — worst embedding max-abs diff **5.439e-05**, worst raw-score diff **1.128e-07** |
| V4 weights export | `tools/export_v4_weights.py` | max\|diff\| **1.21e-08** over 369 validation rows vs archived `raw_score`s |
| YAMNet TFLite conversion | `tools/export_yamnet_tflite.py` | tfhub-vs-tflite mean-embedding max diff **2.861e-06** |
| Kotlin trim vs numpy | `PreprocessingTest` w/ `gold_trim.bin` (23 cases: 7 synthetic + 16 dev) | starts/ends equal; float32 arrays equal bit-for-bit |
| Kotlin V4 score | `V4ClassifierTest` w/ `v4_fixtures.json` | reproduces archived scores |

Gold/fixture files under `app/src/test/resources/` (mirrored in `model-artifacts/gold/`):
`gold/gold_trim.bin` (19,596,023 B), `gold/gold_trim_index.json`, `gold/v4_fixtures.json`,
`model/v4_weights.json`.

## Constraint checklist

| # | Constraint | Verified |
| --- | --- | --- |
| 1 | No INTERNET permission anywhere in the app | Merged manifest lists only `RECORD_AUDIO` (+ framework-internal `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`); no `usesCleartextTraffic`, no network stack in code |
| 2 | No test-split data used for app gold/fixtures/parity | All tools filter `split in (train, validation)`; V4 test split remains unevaluated |
| 3 | No V5 / no retraining / no model changes | Pipeline identical to archived V4 research; only export/serialization moved to tensorflow-lite + JSON weights |
| 4 | Honest UI wording | Screens use "Research prototype", "Not a medical diagnosis"; bands are visualization-only; never claims "you have URTI" |
| 5 | Research outputs protected | `experiments/**` untouched; `model-artifacts/**` added only; no listening-review files modified (0/90 human reviews preserved) |
| 6 | Offline-first | Model + weights in APK assets; local Room history; no analytics/telemetry |

## Build / run notes

- JDK: use `C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot` (previous temp JDK path
  was partially cleaned and is no longer usable).
- SDK: `C:\Users\mansi\AppData\Local\Android\Sdk` (platforms 36/37, build-tools 34/35/36).
- No emulator/device available locally — on-device TFLite behavior was instead proven by the
  Python TFLite parity harness above; instrumented (device) tests are not run here.
- `librosa`/`sklearn` remain unimportable on this machine (AppControl DLL policy); all gold data
  is produced by `tools/app_trim.py` (validated == research reference) and is deterministic
  (`Random(20260921)`).

## Known limitations

- Import via user-selected files only; format coverage depends on MediaCodec (best with
  m4a/aac/wav/ogg in 16 kHz or standard sample rates; caller upsamples/downsamples in code).
- Recording requires the mic permission (runtime request on first use).
- Research score threshold is 0.5 on the V4 validation AUC ~0.529 curve — far below any
  clinical utility; do not interpret as diagnostic.