# Review of shared Kaggle training conversation

Source: https://chatgpt.com/share/6ab14c27-700c-83ee-bcfa-92bdd3ffaba1
Read 2026-09-21. This review uses the shared code and pasted outputs, not a fresh verification of Kaggle artifacts. No new training run was started.

## Reconstructed state

- Termux CLI installation hit native dependency build problems; Debian/proot became the working environment. These installation failures were separate from model performance.
- Original preprocessing rejected UUID `c9f8ed02-90a1-4a0f-9f2c-d3fd04f3ffbd` for too little audible content. The later notebook excluded only that training-set negative recording.
- Updated cohort: 2,465 recordings; training 1,724 (486 positive), validation 369 (104 positive), test 372 (105 positive). Existing split assignments were preserved.
- CNN rerun trained for 13 epochs, choosing epoch 5 by validation AUC (0.55936). It failed afterward because the temporary working folder lacked `dataset_summary.json`. The patch also omitted `predict.py` and `README.md`, which the original export loop requires. A robust fix must copy/check all export dependencies before training; repairing these does not require retraining an existing checkpoint.
- V2 and V3 completed training. A V3 artifact download was interrupted despite COMPLETE status; downloaded metrics do not establish that every large artifact is intact.
- The conversation ends with V4 commands, not V4 results. V4 was intended to run on CPU in Debian, reusing V2 embeddings.

## Results shown in the conversation

| Experiment | ROC AUC | AP | Sensitivity at 0.5 | Specificity at 0.5 | Brier score |
|---|---:|---:|---:|---:|---:|
| V1 small CNN | 0.49067 | 0.27351 | 0.0000 | 1.0000 | 0.20341 |
| V2 YAMNet mean + dense head | 0.58145 | 0.36121 | 0.34286 | 0.75655 | 0.23934 |
| V3 YAMNet mean/max/std + dense head | 0.52991 | 0.30662 | 0.18095 | 0.85393 | 0.23585 |

CNN predicted every test recording negative, yielding 71.77% accuracy simply from class imbalance. V2 detected 36/105 positives and missed 69. V3 detected 19/105 and missed 86. V2 has the highest observed ranking performance of these runs; it is not demonstrated suitable for app screening.

## Corrections and limitations

1. AUC 0.581 is not 58.1% accuracy or proof of a clinically useful signal. Compare paired predictions and uncertainty intervals before calling differences statistically established.
2. V2 improved ranking but its Brier score worsened. At the training positive rate of 486/1724, a constant prevalence predictor has test Brier loss about 0.203. V2 and V3 are worse on this probability-error metric. Class-balanced training changes the objective; scores should not be presented as calibrated disease probabilities.
3. V3 changed pooling, normalization, hidden-layer sizes, dropout, L2 regularization and learning rate simultaneously. Its poorer result cannot be attributed to pooling alone.
4. Matching class proportions across splits does not prove that the split is free of participant, recording-device or site confounding. UUID/hash separation guards recording identity and exact byte duplicates, not participant identity or acoustically duplicated files in different encodings.
5. The same test set has now influenced experiment choices. It remains useful as an exploratory benchmark but is no longer a pristine confirmatory evaluation. Re-randomizing the already inspected cohort does not erase that history. Use development-only selection and seek a genuinely independent final cohort.
6. V4 fits its scaler only on training data during C selection, then refits on train+validation. This is a legitimate final fitting strategy, but unlike V2/V3 its final classifier receives 2,093 examples rather than 1,724. A direct performance comparison must disclose that difference or include a matched-training-size variant.
7. V4 exports a scaler/classifier joblib, not an end-to-end audio model. Deployment also needs the correct YAMNet artifact, matching waveform preparation/mean pooling, package versions, and an inference equivalence check. The original CNN prediction script cannot be used with a YAMNet classifier.
8. Most cohort labels are individual expert cough assessments, not confirmed clinical URTI diagnoses. Better fitting of these labels does not establish throat-infection diagnosis.

## Recommended continuation

First retrieve and preserve V2's manifest, embeddings, classifier, run configuration and predictions; check file integrity and row ordering. Determine whether V4 has already run before launching anything. If no V4 results exist, its regularized linear classifier is a reasonable inexpensive next diagnostic experiment, not a promised accuracy improvement. Select its settings using development data only, record convergence and uncertainty, and label any reused-test reporting exploratory. Keep uncalibrated scores out of a medical-probability claim.
