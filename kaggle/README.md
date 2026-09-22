# URTI cough research training package

This package trains and compares several experimental cough-audio classifiers on Kaggle GPU. It is not a reproduction of Mele0/COUGHVID's reported accuracy. Labels are expert assessments of cough recordings, not independently confirmed diagnoses.

## Kaggle steps

1. Upload `urti_kaggle_dataset.zip` as a private Kaggle dataset and let Kaggle unpack it. Do not upload the full original ZIP as well.
2. Import `URTI_Training.ipynb` into a Kaggle notebook. Attach the private dataset as input.
3. Select a GPU accelerator. Run the notebook cells in order. Setup checks dependencies and installs only missing packages; that step needs Internet if anything is missing.
4. The notebook finds the attached `manifest.csv` automatically. If multiple matching datasets are attached, set `DATA_ROOT` to the correct folder.
5. Preprocessing runs on CPU and can take time before GPU training begins. Failed audio decoding stops training and writes a failure report rather than silently changing the cohort.
6. When finished, download `/kaggle/working/urti_model_bundle.zip`. It contains the selected `.keras` model, `urti_model.tflite`, validation-only calibration metadata, configuration, prediction and training code, package versions, metrics, test predictions and evaluation plot.

## Cohort and splitting

Read original `metadata_compiled.csv` from the official ZIP. Keep recordings with at least one of the five recognized expert diagnoses, consistent binary URTI/non-URTI votes, and no expert quality flag of `poor` or `no_cough`. Missing diagnoses are not imputed. Most retained recordings have one expert diagnosis, not multiple-expert consensus. No demographic/location metadata is packaged.

Split approximately 70% train, 15% validation, 15% test, stratified by class with seed 75. Identical audio file bytes are grouped by SHA-256 so duplicates cannot cross splits. Conflicting labels for identical bytes abort preparation. UUIDs identify recordings, not necessarily people: no participant ID is available in the inspected table. Different encodings or recordings from one person can remain undetected.

The training script compares three CNN variants with light spectrogram augmentation and class weighting. Validation ROC-AUC selects the model and checkpoint. The validation split also fits Platt and isotonic calibration and chooses the decision threshold; test data is used only for final reporting. Avoid repeated model selection using the test report.

## Preprocessing and score interpretation

FFmpeg decodes mono 16 kHz PCM. Reject invalid, silent, <0.2-second or >60-second inputs. Trim at 30 dB below the peak, normalize peak amplitude, centre-crop to eight seconds or pad on the right. Compute 64 mel bands using 512-point FFT, 256-sample hop, Hann window, centred frames, Slaney scale/normalization and power 2. Convert to relative dB limited to 80 dB, then scale to [0,1]. This differs intentionally from the upstream RGB spectrogram workflow.

The module used for training is also shipped for inference. Preprocessing does not establish that an input contains a cough; silence checks cannot reject all speech/noise. Real app use needs cough detection and validation on device recordings before interpreting scores.

The selected bundle reports both a raw sigmoid score and a Platt-calibrated score for the dataset's expert URTI label. Brier score, log loss and a reliability plot assess calibration. Even calibrated scores are dataset-model estimates, not clinical disease probabilities. Non-URTI can include other illnesses.

## After downloading the bundle

Install versions compatible with its `environment.txt`, and install FFmpeg on the inference machine. In the extracted bundle directory:

```text
python predict.py recording.webm --bundle .
```

For Android integration, start with `urti_model.tflite`, `preprocessing.json`, and `platt_calibration.json` from the downloaded bundle. The current Android app uses a separate YAMNet+V4 pipeline, so replacing it with this spectrogram CNN requires adding the same mel-spectrogram preprocessing on device.

Source dataset: https://zenodo.org/records/7024894
Reference project: https://github.com/Mele0/COUGHVID
API references: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html and https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed

Local verification covers packaging, cohort counts, split isolation and Python/notebook syntax. TensorFlow training and audio preprocessing must be exercised in Kaggle; the local machine does not have their required libraries installed.
