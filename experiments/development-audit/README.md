# Development data audit — 2026-09-21

Scope: 2,093 training/validation recordings only; no test audio or labels were analyzed in this audit. Code: `../audit_development.py`. All checks ran locally; no GPU training.

## Findings

- 590 URTI-positive and 1,503 non-URTI development recordings.
- Every one of the 590 positives has exactly one available expert diagnosis. Among negatives, 1,488 have one diagnosis and 15 have four. Thus the current positive cohort has no multi-expert corroboration. This does not prove those labels wrong, but limits confidence in them. Restricting to multi-expert agreement would eliminate the entire development positive class.
- Diagnosis annotation columns have different URTI vote rates: column 1 has 20 positive versus 458 negative votes (~4.2% positive); column 2 has 102 versus 421 (~19.5%); column 3 has 253 versus 320 (~44.2%); column 4 has 215 versus 349 (~38.1%). Column identity has not been independently verified as stable physician identity. Differences could reflect assessor variation, assigned case mix, or both; they do not establish bias or cause weak model performance.
- Existing `cough_detected` scores are below 0.5 for 70 recordings (19 positive, 51 negative), and below 0.8 for 244 (56 positive, 188 negative). These model scores are audit flags, not proof of non-cough audio. All selected expert quality votes are good/ok by cohort construction.
- A fixed random sample of 48 recordings (12 per development split/class) decoded successfully. None had >1% clipped samples or <0.2-second active span by the audit's energy rule. Median duration was 9.78 seconds, range 2.64–10.01. This is an automated signal-quality spot check, not listening or verification of cough content. It does not rule out speech, environmental noise, or problems elsewhere in the cohort.

## Deliverables

- `results.json`: aggregate statistics and limitations.
- `labels.csv`: development annotation counts and cough scores.
- `audio_spot_check.csv`: reproducible 48-recording sample and measured quality.
- `review_queue.csv`: 70 low-cough-score recordings, ordered for review. No labels or split assignments were changed.

## Conclusion

The checked audio is readable, but the audit identifies uncertainty in label corroboration and potentially non-cough content. Neither issue is proven to explain the weak models. Do not train on annotation identity to inflate apparent performance, automatically relabel recordings, or remove cases merely because a model predicts them poorly.

Before another expensive training run, review cough content in the flagged development recordings and a random comparison sample, ideally without revealing diagnosis labels to the reviewer. Document audio-only exclusion rules in advance. Verify annotation-column meaning with the original data documentation. Seek independently assessed positives if diagnostic claims are the goal. Any next representation/segmentation experiment should use development-only model selection and report these limitations.
