# COUGHVID Dataset Analysis Report

**Date:** 2026-09-20  
**Dataset:** COUGHVID public_dataset_v3 (official v3 from Zenodo)  
**Source:** https://zenodo.org/records/7024894

---

## 1. Original Expert Diagnosis Fields (metadata_preprocessed.csv)

The `metadata_preprocessed.csv` contains 3,268 rows across 2,545 unique UUIDs. Each row represents a recording segment with annotations from 4 physicians.

### diagnosis_upper_infection Distribution (Expert Aggregated Label)

| Value | Count | Interpretation |
|-------|-------|----------------|
| 0.0   | 2,172 | Clear Non-URTI |
| 1.0   | 788   | Clear URTI |
| 0.2   | 130   | Fractional/Uncertain |
| 0.4   | 92    | Fractional/Uncertain |
| 0.6   | 29    | Fractional/Uncertain |
| 0.8   | 7     | Fractional/Uncertain |
| NaN   | 50    | Missing |

**Clean binary labels (0.0 or 1.0 only):** 2,960 rows (788 URTI, 2,172 Non-URTI)  
**Fractional labels (0.2, 0.4, 0.6, 0.8):** 258 rows — represent physician disagreement/uncertainty  
**Missing labels (NaN):** 50 rows

### Per-UUID Consensus (Clean Binary Annotations Only)

Aggregating clean (0.0/1.0) annotations by UUID:

| Consensus | UUID Count |
|-----------|------------|
| URTI (majority 1.0) | 590 |
| Non-URTI (majority 0.0) | 1,686 |
| Tie (equal 0.0 and 1.0) | 25 |
| **Total UUIDs with clean annotations** | **2,301** |

**UUIDs with conflicting clean annotations (both 0.0 and 1.0 present):** 66

### Physician-Level Annotations

Each row has 4 physician annotations (`physician_id_1` through `physician_id_4`). Values are in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, where fractional values indicate that physician's internal uncertainty or sub-annotation.

- Rows with any fractional physician value: 340
- The `diagnosis_upper_infection` column is **NOT** a simple mean of the 4 physicians
- 50 rows have no physician annotations (all NaN)

### Duplicate Recording IDs

- **111 UUIDs** appear more than once (max 4 rows per UUID)
- **834 total duplicate rows** (3,268 - 2,545 = 723 extra rows from duplicates, but some UUIDs have 3-4 rows)
- Duplicate UUIDs often have **conflicting labels** across their rows

---

## 2. Imputed Metadata (metadata_imputed.csv) — NOT USED AS GROUND TRUTH

- 3,280 rows, 2,890 unique UUIDs
- Categorical `diagnosis` field: 895 upper_infection, 757 healthy_cough, 746 lower_infection, 664 COVID-19, 218 obstructive_disease
- **Excluded from ground truth** per audit requirements: imputation pipeline provenance unclear; COVID status and `status_SSL` not substituted for URTI target

---

## 3. Audio Data Availability

| Source | Unique UUIDs |
|--------|--------------|
| metadata_preprocessed.csv | 2,545 |
| processed_audio.csv (MFCC features) | 657 |
| **Official dataset audio (extracted)** | **~29,348 .webm + 3,309 .wav files** |
| Overlap (metadata ∩ audio) | ~2,500+ UUIDs estimated |

The official dataset provides raw audio (.webm/.wav) and per-recording JSON metadata for ~34,434 recordings.

---

## 4. Label Policy for Training

**Ground truth definition:** Use only `diagnosis_upper_infection` from `metadata_preprocessed.csv` with clean binary values (0.0 = Non-URTI, 1.0 = URTI).

**Excluded:**
- Fractional labels (0.2, 0.4, 0.6, 0.8) — ambiguous
- NaN labels (50 rows) — missing
- Imputed diagnoses — not original expert annotations
- COVID status / `status_SSL` — different clinical target

**Conflict resolution:** For UUIDs with multiple clean annotations, use majority vote. Ties (25 UUIDs) excluded. UUIDs with only fractional/NaN annotations excluded.

**Estimated usable UUIDs with audio:**
- Clean binary consensus URTI: ~590 UUIDs (subset with audio)
- Clean binary consensus Non-URTI: ~1,686 UUIDs (subset with audio)
- After audio availability filter: estimated 400-500 URTI, 1000-1300 Non-URTI

---

## 5. Key Differences from Repository

| Aspect | Repository | This Pipeline |
|--------|------------|---------------|
| Label source | `labels_new.csv` (external, unverified) | `metadata_preprocessed.csv` clean binary only |
| Split strategy | Over augmented images (leakage risk) | **Split by source UUID before augmentation** |
| Spectrogram params | Undocumented; 225.0 normalization | Documented; 255.0 normalization |
| Image shape | (88,39,3) vs (39,88,3) confusion | Fixed: (height=128, width=128, channels=1) |
| Architecture | Multiple (LSTM, CNN-LSTM, RF, ensemble) | Single reproducible CNN-LSTM |
| Evaluation | Thresholded predictions for AUC | **Probability-based ROC AUC, calibration** |
| Test set | Not held out from model selection | **Final untouched test set** |

---

## 6. Unresolved Issues

1. **Audio format:** .webm files need conversion to .wav for consistent processing
2. **Segment-level vs recording-level:** metadata_preprocessed.csv has segments; JSON metadata is per-recording. Need to map segments to audio files.
3. **Participant IDs:** Not available in preprocessed metadata; cannot group by participant for split
4. **Class imbalance:** ~1:3 URTI:Non-URTI ratio; will need class weighting or sampling
5. **Fractional labels:** 258 rows with fractional labels — currently excluded; could explore soft labels
6. **Missing physician annotations:** 50 rows with all-NaN physicians — excluded
7. **Tie UUIDs:** 25 UUIDs with equal 0.0/1.0 votes — excluded from training

---

## 7. Next Steps

1. Convert .webm audio to .wav (standardize sample rate, mono)
2. Parse JSON metadata to link audio files to UUIDs
3. Join with metadata_preprocessed.csv on UUID
4. Apply label policy → create clean dataset
5. Split by UUID (stratified) → train/val/test
6. Generate mel-spectrograms with documented parameters
7. Train CNN-LSTM with identical train/inference preprocessing
8. Export .keras model + preprocessing config + label mapping + prediction script
9. Evaluate on held-out test set (ROC AUC, sensitivity, specificity, calibration)