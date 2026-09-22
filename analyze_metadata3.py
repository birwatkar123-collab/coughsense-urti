import pandas as pd
import numpy as np

df = pd.read_csv('D:/urti/reference/COUGHVID/Data/metadata_preprocessed.csv')

phys_cols = ['physician_id_1', 'physician_id_2', 'physician_id_3', 'physician_id_4']

# Check actual unique values in physician columns
print('=== PHYSICIAN COLUMN UNIQUE VALUES ===')
for col in phys_cols:
    vals = sorted(df[col].dropna().unique())
    print(f'{col}: {vals[:20]}...' if len(vals) > 20 else f'{col}: {vals}')
print()

# The fractional values 0.2, 0.4, 0.6, 0.8 suggest these are already aggregated
# Let's check: are these the mean of multiple annotations per physician?
# Or are they individual physician scores?

# Let's look at a few rows in detail
print('=== SAMPLE ROWS WITH PHYSICIAN VALUES ===')
for i in range(5):
    row = df.iloc[i]
    print(f'Row {i}: uuid={row["uuid"]}')
    for col in phys_cols:
        print(f'  {col}: {row[col]}')
    print(f'  diagnosis_upper_infection: {row["diagnosis_upper_infection"]}')
    print()

# The diagnosis_upper_infection values match the pattern of physician values
# 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 - these look like proportions (k/5 or similar)
# But there are 4 physicians... so maybe it's mean of 5 annotations per physician?
# Or maybe each physician gives a score 0-5?

# Let's check the imputed metadata for more clues
df_imp = pd.read_csv('D:/urti/reference/COUGHVID/Data/metadata_imputed.csv')
print('=== IMPUTED METADATA DIAGNOSIS DISTRIBUTION ===')
print(df_imp['diagnosis'].value_counts())
print()

# Check overlap between preprocessed and imputed
pre_uuids = set(df['uuid'].unique())
imp_uuids = set(df_imp['uuid'].unique())
print(f'Preprocessed UUIDs: {len(pre_uuids)}')
print(f'Imputed UUIDs: {len(imp_uuids)}')
print(f'Intersection: {len(pre_uuids & imp_uuids)}')
print(f'Only in preprocessed: {len(pre_uuids - imp_uuids)}')
print(f'Only in imputed: {len(imp_uuids - pre_uuids)}')
print()

# For intersection, compare diagnosis
intersection = pre_uuids & imp_uuids
sample = list(intersection)[:10]
for uuid in sample:
    pre_row = df[df['uuid'] == uuid].iloc[0]
    imp_row = df_imp[df_imp['uuid'] == uuid].iloc[0]
    print(f'{uuid}: pre diag_upper={pre_row["diagnosis_upper_infection"]}, imp diag={imp_row["diagnosis"]}, imp status={imp_row["status"]}')