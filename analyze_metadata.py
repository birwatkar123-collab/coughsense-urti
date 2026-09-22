import pandas as pd

# Comprehensive analysis of metadata_preprocessed.csv
df = pd.read_csv('D:/urti/reference/COUGHVID/Data/metadata_preprocessed.csv')

print('=== DATASET OVERVIEW ===')
print(f'Total rows: {len(df)}')
print(f'Unique UUIDs: {df["uuid"].nunique()}')
print()

print('=== DIAGNOSIS_UPPER_INFECTION (expert annotations) ===')
print(df['diagnosis_upper_infection'].value_counts(dropna=False).sort_index())
print()

# Clean binary labels: only 0.0 and 1.0 are clear expert annotations
clean_binary = df[df['diagnosis_upper_infection'].isin([0.0, 1.0])]
print(f'Rows with clean binary labels (0.0 or 1.0): {len(clean_binary)}')
print(f'  URTI (1.0): {(clean_binary["diagnosis_upper_infection"] == 1.0).sum()}')
print(f'  Non-URTI (0.0): {(clean_binary["diagnosis_upper_infection"] == 0.0).sum()}')
print()

# Fractional/conflicting labels
fractional = df[df['diagnosis_upper_infection'].isin([0.2, 0.4, 0.6, 0.8])]
print(f'Rows with fractional labels (0.2, 0.4, 0.6, 0.8): {len(fractional)}')
print(fractional['diagnosis_upper_infection'].value_counts().sort_index())
print()

# Missing labels
missing = df[df['diagnosis_upper_infection'].isna()]
print(f'Rows with missing labels (NaN): {len(missing)}')
print()

# Per-UUID analysis for clean labels
clean_uuids = clean_binary.groupby('uuid')['diagnosis_upper_infection'].apply(list).reset_index()
clean_uuids['n_annotations'] = clean_uuids['diagnosis_upper_infection'].apply(len)
clean_uuids['n_urti'] = clean_uuids['diagnosis_upper_infection'].apply(lambda x: sum(x))
clean_uuids['n_non_urti'] = clean_uuids['n_annotations'] - clean_uuids['n_urti']
clean_uuids['consensus'] = clean_uuids.apply(lambda r: 'URTI' if r['n_urti'] > r['n_non_urti'] else ('Non-URTI' if r['n_non_urti'] > r['n_urti'] else 'Tie'), axis=1)

print('=== PER-UUID CONSENSUS (clean binary annotations only) ===')
print(f'Unique UUIDs with clean annotations: {len(clean_uuids)}')
print(f'Consensus URTI: {(clean_uuids["consensus"] == "URTI").sum()}')
print(f'Consensus Non-URTI: {(clean_uuids["consensus"] == "Non-URTI").sum()}')
print(f'Tie: {(clean_uuids["consensus"] == "Tie").sum()}')
print()

# Conflicting annotations per UUID
conflicts = clean_uuids[(clean_uuids['n_urti'] > 0) & (clean_uuids['n_non_urti'] > 0)]
print(f'UUIDs with conflicting clean annotations: {len(conflicts)}')
if len(conflicts) > 0:
    print(conflicts[['uuid', 'n_annotations', 'n_urti', 'n_non_urti', 'consensus']].to_string())
print()

# Duplicate UUIDs overall
dup_counts = df['uuid'].value_counts()
print(f'UUIDs appearing >1 time: {(dup_counts > 1).sum()}')
print(f'Max rows per UUID: {dup_counts.max()}')
print()

# Physician agreement analysis
phys_cols = ['physician_id_1', 'physician_id_2', 'physician_id_3', 'physician_id_4']
print('=== PHYSICIAN ANNOTATION AGREEMENT ===')
df['n_physicians'] = df[phys_cols].notna().sum(axis=1)
df['phys_sum'] = df[phys_cols].fillna(0).sum(axis=1)
print(f'Rows by number of physician annotations:')
print(df['n_physicians'].value_counts().sort_index())
print()

# Rows where all 4 physicians agree on 0 or 1
df_4phys = df[df['n_physicians'] == 4].copy()
df_4phys['all_zero'] = (df_4phys[phys_cols] == 0.0).all(axis=1)
df_4phys['all_one'] = (df_4phys[phys_cols] == 1.0).all(axis=1)
print(f'Rows with 4 physicians: {len(df_4phys)}')
print(f'  All 4 say 0 (Non-URTI): {df_4phys["all_zero"].sum()}')
print(f'  All 4 say 1 (URTI): {df_4phys["all_one"].sum()}')
print(f'  Disagreement among 4: {len(df_4phys) - df_4phys["all_zero"].sum() - df_4phys["all_one"].sum()}')