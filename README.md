# CoughSense URTI

CoughSense URTI is an offline Android research prototype for cough-audio analysis. It records or imports a cough sound, preprocesses the audio on device, runs an ML pipeline, and stores a local research score history.

Important: this project is **not a medical device** and does **not diagnose URTI**. Scores are experimental model outputs from labeled cough-audio research data.

## What It Does

- Records cough audio on Android with microphone permission.
- Imports audio files supported by Android MediaCodec.
- Processes audio fully offline; there is no app `INTERNET` permission.
- Runs an on-device YAMNet embedding model plus a V4 classifier.
- Shows a research model score, audio quality information, and local history.
- Includes a Kaggle GPU training package for comparing and calibrating newer experimental CNN models.

## ML Pipeline

The shipped Android app uses:

1. Decode or record mono 16 kHz audio.
2. Reject invalid, silent, too-short, or too-long audio.
3. Trim silence with a deterministic Kotlin port matching the research preprocessing.
4. Run a dynamic-input YAMNet TFLite model.
5. Mean-pool the audio embeddings.
6. Apply exported V4 classifier weights.
7. Compare the score with the reference threshold.

The app reports a research score only. A score above threshold means the cough resembles patterns learned from the research labels; it is not proof of illness.

## Benefits

- **Offline:** works after installation without internet.
- **Private:** recordings and history stay on the phone.
- **Fast:** gives a model score in seconds.
- **Repeatable:** local history helps compare recordings over time.
- **Research-focused:** includes validation scripts, Kaggle training code, and reproducibility notes.

## Repository Layout

- `app/` - Android app project.
- `kaggle/` - Kaggle dataset packaging and GPU training notebook/scripts.
- `tools/` - export, parity, and preprocessing validation tools.
- `experiments/` - research audits and validation outputs.
- `assets/promo/` - promotional schematic image.
- `APP-STATUS.md` - Android delivery and verification notes.

## Build The Android App

Requirements:

- JDK 21
- Android SDK
- Gradle wrapper included in `app/`

```powershell
cd app
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
```

Debug APK:

```text
app/app/build/outputs/apk/debug/app-debug.apk
```

Install on a connected Android phone:

```powershell
adb install -r app\app\build\outputs\apk\debug\app-debug.apk
```

## Kaggle Training

The `kaggle/` folder contains a prepared workflow to train and compare experimental cough classifiers on Kaggle GPU:

- compares multiple CNN variants
- uses class weighting and light augmentation
- selects by validation ROC-AUC
- fits Platt/isotonic calibration on validation data
- reports held-out test metrics
- exports `.keras`, `.tflite`, metrics, predictions, and calibration metadata

See `kaggle/README.md` for details.

## Safety Notice

This project is for research and prototyping only. It should not be used to make medical decisions, replace clinical testing, or determine whether someone has or does not have URTI. If a person has symptoms or health concerns, they should consult a qualified healthcare professional.

## Credits

- **COUGHVID dataset:** Lara Orlandic, Tomas Teijeiro, and David Atienza. COUGHVID public cough dataset, Zenodo. https://zenodo.org/records/7024894
- **Reference project:** Mele0/COUGHVID research code and related cough-audio workflow ideas. https://github.com/Mele0/COUGHVID
- **YAMNet:** TensorFlow Hub YAMNet audio embedding model by Google/TensorFlow. https://tfhub.dev/google/yamnet/1
- **TensorFlow Lite:** on-device model runtime used by the Android app.
- **Android/Kotlin/Jetpack Compose:** app platform and UI framework.
- **Kaggle:** free GPU runtime used for training experiments.
- **Project development:** birwatkar123-collab, with AI-assisted coding and documentation support.

## License And Data

Dataset-derived materials must follow the original COUGHVID licensing and attribution requirements. Model outputs and this prototype are provided for research use only. Add a project license before using this code in production or redistributing it outside research/prototyping contexts.
