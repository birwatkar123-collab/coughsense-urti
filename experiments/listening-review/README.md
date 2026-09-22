# Blinded audio review

Open `index.html` in a browser, keeping its `audio` subfolder beside it. The review contains 70 development recordings flagged by the existing cough score plus 20 randomly selected unflagged development comparison recordings, shuffled with seed 20260921. Audio totals about 12.5 minutes, excluding rating time. No test recordings are included.

Rate audible cough content only, not infection. Use uncertain when needed. Download the review JSON before leaving. Ratings remain in browser storage when available; nothing is uploaded. The private mapping is outside this folder in `../listening-review-key.csv`; do not inspect it until ratings are complete. No recordings have yet been manually rated. A listener is required; automated signal checks were not presented as listening judgments.

All 90 mono 16-kHz WAV files were verified readable and the JavaScript passed syntax checking. Full interactive browser behavior has not been tested.

## Annotation mapping result

Compared every diagnosis_1 through diagnosis_4 field for all 2,093 development recordings against the corresponding original JSON expert_labels_1 through expert_labels_4 diagnosis entry: 8,372 comparisons, zero mismatches. Thus the CSV suffixes preserve the original expert-label slots; no column-remapping error was found. This does not establish correctness of clinical labels or explain differences in expert case mix.

The original paper describes numbered expert-label dictionaries and reports limited diagnostic agreement. Source: https://infoscience.epfl.ch/server/api/core/bitstreams/78d43390-9ee4-4fbe-8ca2-34ccfeccfae6/content (COUGHVID, Scientific Data, 2021, DOI 10.1038/s41597-021-00937-4). It predates the v3 archive, so actual v3 mapping was checked directly against its JSON files.

After listening, review flagged-versus-comparison counts before deciding any exclusions. These audio-content ratings cannot establish or change URTI diagnosis. No exclusion, relabeling or new training was performed.
