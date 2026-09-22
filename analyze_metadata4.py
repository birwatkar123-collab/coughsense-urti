import pandas as pd
import numpy as np

df = pd.read_csv('D:/urti/reference/COUGHVID/Data/metadata_preprocessed.csv')

phys_cols = ['physician_id_1', 'physician_id_2', 'physician_id_3', 'physician_id_4']

# Find rows where physician columns have fractional values
for col in phys_cols:
    frac_rows = df[(df[col] > 0) & (df[col] < 1)]
    if len(frac_rows) > 0:
        print(f'{col} fractional rows: {len(frac_rows)}')
        print(frac_rows[[col, 'diagnosis_upper_infection']].head(10))
        print()

# Let's check all physician values across all rows
all_phys_vals = []
for col in phys_cols:
    all_phys_vals.extend(df[col].dropna().tolist())

print('All physician value counts:')
vals, counts = np.unique(all_phys_vals, return_counts=True)
for v, c in zip(vals, counts):
    print(f'  {v}: {c}')
print()

# Check if diagnosis_upper_infection = mean of physician_id_1..4
df['phys_mean'] = df[phys_cols].mean(axis=1)
df['match'] = np.isclose(df['diagnosis_upper_infection'], df['phys_mean'], atol=0.01)
print(f'diagnosis_upper_infection == mean(physicians): {df["match"].sum()} / {len(df)}')
mismatch = df[~df['match']]
if len(mismatch) > 0:
    print('Mismatches:')
    print(mismatch[phys_cols + ['diagnosis_upper_infection', 'phys_mean']].head(20))
print()

# The fractional values in physician columns must come from somewhere
# Let's check if there are rows where physician columns have fractional values
frac_any = df[
    ((df['physician_id_1'] > 0) & (df['physician_id_1'] < 1)) |
    ((df['physician_id_2'] > 0) & (df['physician_id_2'] < 1)) |
    ((df['physician_id_3'] > 0) & (df['physician_id_3'] < 1)) |
    ((df['physician_id_4'] > 0) & (df['physician_id_4'] < 1))
]
print(f'Rows with any fractional physician value: {len(frac_any)}')
if len(frac_any) > 0:
    print(frac_any[phys_cols + ['diagnosis_upper_infection']].head(20))