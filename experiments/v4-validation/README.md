# V4 results — 2026-09-21

Ran locally on CPU using the downloaded V2 YAMNet mean embeddings. No Kaggle training run or GPU quota was used. The Kaggle notebook listing contained V1/V2/V3, not V4; a prior phone-local V4 cannot be ruled out by that listing.

## Method

Matched all 2,465 manifest IDs, labels, split assignments and source hashes to the original prepared cohort after the documented single exclusion. Verified finite embeddings of shape (2465,1024). The V2 code preserves manifest order when saving embeddings; hashes of downloaded files are recorded in results.json.

Fit a standardized, class-balanced logistic regression. Seven C values (0.001 through 1.0) were compared using five-fold CV within the 1,724 training recordings; each fold fits its own scaler. Selected C=0.001, mean training-CV AUC=0.52775. Sigmoid calibration used cross-fitted training predictions. Both raw and calibrated models were then evaluated on the same 369 validation recordings. No test predictions or test metrics were computed.

Unlike the shared-chat proposal, this version selects settings within training, preserves validation for reporting, and does not refit on train+validation or evaluate the previously inspected test set. Its numbers must not be compared directly to earlier test numbers as a matched evaluation.

## Findings

| Validation measure | Raw logistic model | After sigmoid calibration | Constant training prevalence |
|---|---:|---:|---:|
| ROC AUC | 0.52935 | 0.52935 | 0.50000 |
| Average precision | 0.30343 | 0.30343 | 0.28184 |
| Brier score (lower is better) | 0.25250 | 0.20225 | 0.20241 |
| Sensitivity at 0.5 | 55.77% | 0% | 0% |
| Specificity at 0.5 | 50.19% | 100% | 100% |

Raw classification detected 58/104 positives but falsely flagged 132/265 negatives. After calibration all scores were below 0.5, so all recordings were classified negative at that fixed threshold. Calibration changes score interpretation, not the underlying ranking here.

Stratified bootstrap (1,000 resamples) gives validation AUC 95% interval 0.46437–0.59123. The calibrated-minus-constant Brier difference interval is -0.001429–0.001231, including zero. These intervals condition on the fitted model and do not include training/model-selection variability; the validation cohort has also been used in prior experiments. They are exploratory uncertainty estimates, not certification of clinical performance.

## Conclusion and next decision

This linear classifier does not establish useful URTI discrimination. Calibration largely moves estimates toward the class prevalence, without demonstrating meaningful individualized probability improvement. Do not deploy it as an infection detector or describe its scores as validated medical probabilities.

Before another expensive model run, audit expert-label consistency and cough-versus-background content, and define a development evaluation protocol. If continuing with acoustic models, use a controlled representation comparison on development data; acquire independently labeled evaluation recordings for a final claim. Lowering the threshold alone does not create new discriminatory information.

## Artifacts and verification

- `results.json`: full grid, validation metrics, intervals, calibration bins and source hashes.
- `validation_predictions.csv`: UUID-aligned raw and calibrated scores.
- `v4_models.joblib`: standardized raw classifier and calibrated classifier; requires compatible scikit-learn (run used 1.9.1).
- `../run_v4.py`: reproducible experiment code. It refuses to overwrite a completed results file.

Export/reload predictions matched numerically. These models accept 1,024-D V2 embeddings, not raw audio. Matching YAMNet and waveform preprocessing remain necessary for any audio inference use. Do not load untrusted joblib files.
