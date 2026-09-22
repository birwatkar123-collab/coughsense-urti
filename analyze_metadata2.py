import pandas as pd
import numpy as np

# Comprehensive analysis of metadata_preprocessed.csv
df = pd.read_csv('D:/urti/reference/COUGHVID/Data/metadata_preprocessed.csv')

phys_cols = ['physician_id_1', 'physician_id_2', 'physician_id_3', 'physician_id_4']

# Per-UUID analysis using ALL physician annotations (fractional values represent proportion)
# Each row has 4 physician annotations for that recording segment
# The diagnosis_upper_infection column appears to be the mean of the 4 physicians
print('=== PHYSICIAN ANNOTATION DETAILS ===')
# Check if diagnosis_upper_infection = mean of 4 physicians
df['phys_mean'] = df[phys_cols].mean(axis=1)
df['match'] = np.isclose(df['diagnosis_upper_infection'], df['phys_mean'], atol=0.01)
print(f'diagnosis_upper_infection matches physician mean: {df["match"].sum()} / {len(df)}')
print()

# Per-UUID aggregation using physician votes
# Each physician annotation is a vote (0 or 1, but we have fractional which means disagreement)
# Let's count total physician-votes per UUID
uuid_votes = []
for uuid, group in df.groupby('uuid'):
    # Flatten all physician annotations for this UUID
    all_votes = []
    for _, row in group.iterrows():
        for col in phys_cols:
            val = row[col]
            if pd.notna(val):
                all_votes.append(val)
    if len(all_votes) > 0:
        n_votes = len(all_votes)
        n_urti = sum(1 for v in all_votes if v >= 0.5)  # treat >=0.5 as URTI vote
        n_non = n_votes - n_urti
        consensus = 'URTI' if n_urti > n_non else ('Non-URTI' if n_non > n_urti else 'Tie')
        uuid_votes.append({
            'uuid': uuid,
            'n_votes': n_votes,
            'n_urti': n_urti,
            'n_non': n_non,
            'consensus': consensus,
            'frac_urti': n_urti / n_votes
        })

uuid_df = pd.DataFrame(uuid_votes)
print(f'Total UUIDs with physician votes: {len(uuid_df)}')
print(f'Consensus URTI: {(uuid_df["consensus"] == "URTI").sum()}')
print(f'Consensus Non-URTI: {(uuid_df["consensus"] == "Non-URTI").sum()}')
print(f'Tie: {(uuid_df["consensus"] == "Tie").sum()}')
print()

# Conflicts
conflicts = uuid_df[(uuid_df['n_urti'] > 0) & (uuid_df['n_non'] > 0)]
print(f'UUIDs with conflicting physician votes: {len(conflicts)}')
print()

# Show UUIDs with high agreement (>75% or <25%)
high_agree_urti = uuid_df[(uuid_df['frac_urti'] >= 0.75) & (uuid_df['n_votes'] >= 4)]
high_agree_non = uuid_df[(uuid_df['frac_urti'] <= 0.25) & (uuid_df['n_votes'] >= 4)]
print(f'UUIDs with >=75% URTI votes (min 4 votes): {len(high_agree_urti)}')
print(f'UUIDs with <=25% URTI votes (min 4 votes): {len(high_agree_non)}')
print()

# Now check which UUIDs have audio in processed_audio.csv
audio_df = pd.read_csv('D:/urti/reference/COUGHVID/Data/processed_audio.csv')
audio_uuids = set(audio_df['uuid'].unique())
meta_uuids = set(df['uuid'].unique())
print(f'UUIDs in metadata: {len(meta_uuids)}')
print(f'UUIDs in processed_audio: {len(audio_uuids)}')
print(f'UUIDs in both: {len(meta_uuids & audio_uuids)}')
print(f'UUIDs only in metadata: {len(meta_uuids - audio_uuids)}')
print(f'UUIDs only in audio: {len(audio_uuids - meta_uuids)}')
print()

# For UUIDs with high agreement AND audio, how many?
high_agree_uuids = set(high_agree_urti['uuid']) | set(high_agree_non['uuid'])
print(f'High-agreement UUIDs with audio: {len(high_agree_uuids & audio_uuids)}')
print()

# Check duplicate UUIDs in processed_audio
audio_dup = audio_df[audio_df.duplicated(subset=['uuid'], keep=False)]
print(f'Processed audio rows with duplicate UUIDs: {len(audio_dup)}')
print(f'Unique duplicate UUIDs in audio: {audio_dup["uuid"].nunique()}')
print()

# For the clean binary diagnosis_upper_infection (0.0 or 1.0) per UUID
clean = df[df['diagnosis_upper_infection'].isin([0.0, 1.0])]
clean_uuids = clean.groupby('uuid')['diagnosis_upper_infection'].apply(list).reset_index()
clean_uuids['n_annot'] = clean_uuids['diagnosis_upper_infection'].apply(len)
clean_uuids['n_urti'] = clean_uuids['diagnosis_upper_infection'].apply(lambda x: sum(x))
clean_uuids['n_non'] = clean_uuids['n_annot'] - clean_uuids['n_urti']
clean_uuids['consensus'] = clean_uuids.apply(lambda r: 'URTI' if r['n_urti'] > r['n_non'] else ('Non-URTI' if r['n_non'] > r['n_urti'] else 'Tie'), axis=1)

# Intersection with audio
clean_uuids_set = set(clean_uuids['uuid'])
print(f'Clean-label UUIDs with audio: {len(clean_uuids_set & audio_uuids)}')
print()

# Final usable counts: UUIDs with high physician agreement AND audio
usable_urti = high_agree_urti[high_agree_urti['uuid'].isin(audio_uuids)]
usable_non = high_agree_non[high_agree_non['uuid'].isin(audio_uuids)]
print(f'=== FINAL USABLE COUNTS (high agreement + audio) ===')
print(f'URTI: {len(usable_urti)}')
print(f'Non-URTI: {len(usable_non)}')
print(f'Total: {len(usable_urti) + len(usable_non)}')
print()

# Show some examples
print('Example URTI UUIDs:')
print(usable_urti[['uuid', 'frac_urti', 'n_votes']].head(10).to_string())
print()
print('Example Non-URTI UUIDs:')
print(usable_non[['uuid', 'frac_urti', 'n_votes']].head(10).to_string())