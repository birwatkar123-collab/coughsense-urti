# Kaggle Multimodel V2 Result

Kaggle run: `birwatkar/urti-cough-multimodel-calibrated-training`, version 2.

The run completed successfully after excluding one recording that failed the packaged preprocessing gate after silence trim.

## Selected Model

- Selected model: `small_cnn`
- Selection rule: highest validation ROC-AUC
- Calibration: Platt calibration fitted on validation split only
- Decision threshold: `0.2829736205336739`

## Held-Out Test Metrics

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.5869 |
| Average precision | 0.3416 |
| Accuracy | 0.5768 |
| Sensitivity | 0.5238 |
| Specificity | 0.5977 |
| Precision | 0.3395 |
| F1 | 0.4120 |
| Brier score | 0.2026 |
| Log loss | 0.5950 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[159, 107], [50, 55]]`.

## Interpretation

This model is **not strong enough for medical use**. It is only a research artifact for comparison and future iteration. The result is slightly above random discrimination, but still weak. It should not replace the current app model without a deliberate Android preprocessing implementation and more validation.

## Files

- `urti_model.tflite` - selected spectrogram CNN exported from Kaggle.
- `preprocessing.json` - audio preprocessing parameters.
- `platt_calibration.json` - calibration metadata.
- `metrics.json` - full comparison metrics for all model candidates.
- `evaluation.png` - validation AUC, test ROC, and calibration plot.
