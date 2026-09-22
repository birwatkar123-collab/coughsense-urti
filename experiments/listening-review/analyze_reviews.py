"""Post-listening-review analysis for the blinded cough-content review.

Joins the downloaded manual review file (urti-audio-reviews.json) to the
PRIVATE mapping (../listening-review-key.csv) and writes an analysis bundle
under ./analysis/. Ratings describe audible cough content only: they never
change diagnosis labels, never change train/validation/test membership, and
never touch model outputs. No test-set records are expected; the validation
summary flags them as errors if any appear.
"""

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from math import isnan
from pathlib import Path

import numpy as np
import scipy.stats as st

EXPECTED_BATCH = 'urti-audio-review-20260921-v1'
REVIEW_COUNT = 90
RATINGS = ('clear_cough', 'mixed_noisy_cough', 'no_cough', 'uncertain')
VALID_RATINGS = set(RATINGS)
ID_PATTERN = re.compile(r'^R\d{3}$')
EXPECTED_IDS = [f'R{i:03d}' for i in range(1, REVIEW_COUNT + 1)]
EXPECTED_ID_SET = set(EXPECTED_IDS)
ALLOWED_GROUPS = {'flagged', 'comparison'}
ALLOWED_SPLITS = {'train', 'validation', 'test'}


def load_key(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def parse_reviews(data):
    if not isinstance(data, dict):
        raise ValueError('Review file must contain a JSON object.')
    raw = data.get('reviews', data) if 'reviews' in data else data
    entries = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                entries.append({'id': str(key).strip(),
                                'rating': value.get('rating'),
                                'notes': value.get('notes')})
            else:
                entries.append({'id': str(key).strip(), 'rating': value, 'notes': None})
    elif isinstance(raw, list):
        for value in raw:
            if not isinstance(value, dict):
                continue
            ident = value.get('id') or value.get('review_id') or value.get('reviewId')
            entries.append({'id': str(ident).strip() if ident is not None else None,
                            'rating': value.get('rating'),
                            'notes': value.get('notes')})
    else:
        raise ValueError('"reviews" must be an object or a list.')
    return entries


def validate(data, key_rows):
    errors = []
    warnings = []
    batch = data.get('batch') if isinstance(data, dict) else None
    if batch is None:
        warnings.append('Review file carries no batch id.')
    elif batch != EXPECTED_BATCH:
        errors.append(f'Batch mismatch: expected {EXPECTED_BATCH!r}, file says {batch!r}.')

    entries = parse_reviews(data)
    ids = [e['id'] for e in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append(f'Duplicate review IDs in file: {duplicates}')
    malformed = sorted({i for i in ids if not (i and ID_PATTERN.match(i))})
    if malformed:
        errors.append(f'Malformed review IDs: {malformed}')
    unknown = sorted({i for i in ids if (not i) or i not in EXPECTED_ID_SET})
    if unknown:
        errors.append(f'Unknown review IDs (not in R001..R090): {unknown}')
    invalid_ratings = [(e['id'], e['rating']) for e in entries
                       if e['id'] in EXPECTED_ID_SET and e['rating'] not in VALID_RATINGS]
    if invalid_ratings:
        errors.append(f'Invalid rating values: {invalid_ratings}')

    by_key_id = {}
    key_ids = []
    for row in key_rows:
        rid = row.get('review_id')
        key_ids.append(rid)
        by_key_id.setdefault(rid, []).append(row)
    if sorted(key_ids) != EXPECTED_IDS:
        errors.append('Mapping row mismatch: key does not cover exactly R001..R090 '
                      f'(found {len(key_ids)} rows).')
    duplicated_key_ids = sorted({i for i in key_ids if key_ids.count(i) > 1})
    if duplicated_key_ids:
        errors.append(f'Mapping duplicate review IDs: {duplicated_key_ids}')
    uuids = [r.get('uuid') for r in key_rows]
    duplicated_uuids = sorted({u for u in uuids if uuids.count(u) > 1})
    if duplicated_uuids:
        errors.append(f'Mapping duplicate UUIDs: {duplicated_uuids}')
    bad_groups = sorted({r.get('group') for r in key_rows
                         if r.get('group') not in ALLOWED_GROUPS})
    if bad_groups:
        errors.append(f'Invalid group values in mapping: {bad_groups}')
    bad_splits = sorted({r.get('split') for r in key_rows
                         if r.get('split') not in ALLOWED_SPLITS})
    if bad_splits:
        errors.append(f'Invalid split values in mapping: {bad_splits}')
    test_rows = [r['review_id'] for r in key_rows if r.get('split') == 'test']
    if test_rows:
        errors.append(f'Test-set records present in review mapping: {test_rows}')

    reviewed_set = {i for i in ids if i in EXPECTED_ID_SET}
    missing = sorted(EXPECTED_ID_SET - reviewed_set)
    if missing:
        warnings.append(f'Missing {len(missing)} of {REVIEW_COUNT} reviews; analysis is preliminary.')
    unreviewed_extra = sorted(reviewed_set - set(key_ids))
    if unreviewed_extra:
        errors.append(f'Reviewed IDs absent from the mapping key: {unreviewed_extra}')

    one_to_one = True
    joined = []
    for rid in sorted(reviewed_set):
        matches = by_key_id.get(rid, [])
        if len(matches) != 1:
            one_to_one = False
            errors.append(f'Reviewed ID {rid} resolves to {len(matches)} mapping rows (not one-to-one).')
            continue
        row = matches[0]
        joined.append({'review_id': rid, 'uuid': row.get('uuid'),
                       'split': row.get('split'), 'label': row.get('label'),
                       'group': row.get('group')})

    return {'batch': batch, 'batch_ok': (batch == EXPECTED_BATCH), 'errors': errors,
            'warnings': warnings, 'entries': entries, 'reviewed_ids': sorted(reviewed_set),
            'missing': missing, 'unknown': unknown, 'duplicates': duplicates,
            'invalid_ratings': invalid_ratings, 'test_rows': test_rows, 'joined': joined,
            'preliminary': bool(missing), 'review_count': len(joined),
            'one_to_one': one_to_one}


def attach_ratings(validated):
    entries_by_id = {}
    for e in validated['entries']:
        entries_by_id.setdefault(e['id'], []).append(e)
    rows = []
    for j in validated['joined']:
        matches = entries_by_id.get(j['review_id'], [])
        entry = matches[0] if matches else {}
        rating = entry.get('rating') if isinstance(entry, dict) else None
        notes = entry.get('notes') if isinstance(entry, dict) else None
        rows.append({**j, 'rating': rating, 'notes': notes,
                     'usable_cough': rating in ('clear_cough', 'mixed_noisy_cough'),
                     'high_quality_cough': rating == 'clear_cough'})
    return rows


def pct(n, total):
    return 0.0 if not total else n / total * 100.0


def group_block(rows):
    n = len(rows)
    counts = OrderedDict((rating, sum(1 for r in rows if r['rating'] == rating))
                         for rating in RATINGS)
    usable = sum(1 for r in rows if r['usable_cough'])
    high = sum(1 for r in rows if r['high_quality_cough'])
    return {'n': n,
            'counts': dict(counts),
            'pct': {rating: pct(counts[rating], n) for rating in RATINGS},
            'usable_cough': {'count': usable, 'pct': pct(usable, n)},
            'high_quality_cough': {'count': high, 'pct': pct(high, n)}}


def fnum(value):
    if value is None or (isinstance(value, float) and isnan(value)):
        return None
    return float(value)


def fisher_block(table):
    out = {'table': table, 'oddsratio': None, 'p_value': None, 'ci95': None, 'note': None}
    a, b = table[0]
    c, d = table[1]
    if a + c == 0 or b + d == 0 or a + b + c + d == 0:
        out['note'] = 'No data across both groups; test not computed.'
        return out
    or_value, p_value = st.fisher_exact(table, alternative='two-sided')
    out['oddsratio'] = fnum(or_value)
    out['p_value'] = fnum(p_value)
    try:
        ci = st.contingency.odds_ratio(table).confidence_interval(confidence_level=0.95)
        out['ci95'] = [float(ci.low), float(ci.high)] if ci.low is not None else None
    except Exception as exc:
        out['ci95'] = None
        out['note'] = f'Confidence interval unavailable: {type(exc).__name__}'
    return out


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def phase5_block(rows):
    usable = [r for r in rows if r['usable_cough']]
    not_usable = [r for r in rows if not r['usable_cough']]
    flagged = [r for r in rows if r['group'] == 'flagged']
    comparison = [r for r in rows if r['group'] == 'comparison']

    def scored(rs):
        return [r for r in rs if r.get('cough_detected') is not None]

    block = {'field': 'cough_detected',
             'source': 'experiments/development-audit/labels.csv (non-diagnostic model score)',
             'n_with_score': len(scored(rows)),
             'usable_median': median([r['cough_detected'] for r in scored(usable)]),
             'not_usable_median': median([r['cough_detected'] for r in scored(not_usable)]),
             'usable_n': len(scored(usable)),
             'not_usable_n': len(scored(not_usable)),
             'flagged_n': len(scored(flagged)),
             'comparison_n': len(scored(comparison)),
             'flagged_median': median([r['cough_detected'] for r in scored(flagged)]),
             'comparison_median': median([r['cough_detected'] for r in scored(comparison)]),
             'spearman': None,
             'mann_whitney': None,
             'caveat': 'The flagged group is defined by this same score (< 0.5), so its lower '
                       'score is mechanical; manual ratings decide whether the flag matches what '
                       'humans hear.'}
    if len(scored(rows)) >= 3 and len({r['cough_detected'] for r in scored(rows)}) >= 2:
        x = np.array([r['cough_detected'] for r in scored(rows)])
        y = np.array([1 if r['usable_cough'] else 0 for r in scored(rows)])
        rho, p = st.spearmanr(x, y)
        block['spearman'] = {'rho': fnum(rho), 'p_value': fnum(p)}
    if len(scored(usable)) >= 1 and len(scored(not_usable)) >= 1:
        u_stat, u_p = st.mannwhitneyu(
            [r['cough_detected'] for r in scored(usable)],
            [r['cough_detected'] for r in scored(not_usable)],
            alternative='two-sided')
        block['mann_whitney'] = {'u_statistic': fnum(u_stat), 'p_value': fnum(u_p)}
    return block


def build_recommendations(groups, fisher_usable, preliminary):
    bullets = []
    if preliminary:
        bullets.append('Analysis is preliminary ({} of 90 reviewed). Finish the review and rerun '
                       'before drawing conclusions; comparisons below are descriptive only.'.format(
                           sum(g['n'] for g in groups.values())))
        bullets.append('Any exclusion must be written as a proposed development-only validation '
                       'experiment, not applied silently; the test set stays untouched.')
        return bullets
    flag = groups['flagged']
    comp = groups['comparison']
    flag_no_p = flag['pct']['no_cough']
    comp_no_p = comp['pct']['no_cough']
    flag_clear_p = flag['pct']['clear_cough']
    comp_clear_p = comp['pct']['clear_cough']
    flag_usable_p = flag['usable_cough']['pct']
    comp_usable_p = comp['usable_cough']['pct']
    diff = comp_usable_p - flag_usable_p
    p = fisher_usable['p_value']

    both_clear = flag_clear_p >= 50.0 and comp_clear_p >= 50.0
    both_lacking = flag_no_p >= 40.0 and comp_no_p >= 40.0
    flagged_more_no = flag_no_p >= comp_no_p + 10.0
    flagged_less_usable = diff >= 15.0

    if both_clear:
        bullets.append('Outcome D: most reviewed recordings in both groups contain a clear cough '
                       '({:.1f}% flagged, {:.1f}% comparison). Audio absence does not explain the '
                       'weak classification results; direct attention to the acoustic-label '
                       'relationship, expert-label uncertainty, or model representation.'.format(
                           flag_clear_p, comp_clear_p))
    if both_lacking:
        bullets.append('Outcome C: recordings lacking an audible cough are common in BOTH groups '
                       '({:.1f}% flagged, {:.1f}% comparison). Recommend a broader development-set '
                       'cough-content audit before further model changes.'.format(
                           flag_no_p, comp_no_p))
    if flagged_more_no and flagged_less_usable and not both_lacking:
        bullets.append('Outcome A: flagged recordings have substantially more no-cough / less '
                       'usable audio than the comparison sample (no-cough {:.1f}% vs {:.1f}%, '
                       'usable {:.1f}% vs {:.1f}%, Fisher p={:.4f}). Consider a validation-only '
                       'audio-content quality-gate experiment; do not exclude or relabel anything '
                       'automatically.'.format(flag_no_p, comp_no_p, flag_usable_p, comp_usable_p,
                                               p))
    if not (both_clear or both_lacking or (flagged_more_no and flagged_less_usable)):
        bullets.append('Outcome B: usable-cough proportions are similar across groups '
                       '(flagged {:.1f}%, comparison {:.1f}%). Cough-presence filtering probably '
                       'does not explain weak model performance; shift investigation toward the '
                       'acoustic-label relationship, expert-label uncertainty, or the model '
                       'representation.'.format(flag_usable_p, comp_usable_p))

    bullets.append('Any exclusion must be written as a proposed development-only validation '
                   'experiment, not applied silently; the test set stays untouched.')
    return bullets


def build_summary(rows, validated, key_path, review_path):
    overall = group_block(rows)
    groups = OrderedDict((g, group_block([r for r in rows if r['group'] == g]))
                         for g in ('flagged', 'comparison'))
    flag, comp = groups['flagged'], groups['comparison']
    usable_table = [[flag['usable_cough']['count'], flag['n'] - flag['usable_cough']['count']],
                    [comp['usable_cough']['count'], comp['n'] - comp['usable_cough']['count']]]
    clear_table = [[flag['counts']['clear_cough'], flag['n'] - flag['counts']['clear_cough']],
                   [comp['counts']['clear_cough'], comp['n'] - comp['counts']['clear_cough']]]
    fisher_usable = fisher_block(usable_table)
    fisher_clear = fisher_block(clear_table)
    fisher_usable['definition'] = ('2x2 table = [flagged usable, flagged other; '
                                   'comparison usable, comparison other]; '
                                   'usable_cough = clear_cough or mixed_noisy_cough')
    fisher_clear['definition'] = '2x2 table = [flagged clear, flagged other; comparison clear, comparison other]'
    ph5 = phase5_block(rows)
    recommendations = build_recommendations(groups, fisher_usable, validated['preliminary'])
    summary = OrderedDict([
        ('experiment', 'Blinded cough-content listening review analysis (audio content, not diagnosis)'),
        ('batch', validated['batch']),
        ('batch_expected', EXPECTED_BATCH),
        ('batch_match', validated['batch_ok']),
        ('key_source', str(key_path)),
        ('review_source', str(review_path)),
        ('ratings_allowed', list(RATINGS)),
        ('expected_reviews', REVIEW_COUNT),
        ('completed_reviews', validated['review_count']),
        ('missing_reviews', validated['missing']),
        ('missing_count', len(validated['missing'])),
        ('preliminary', validated['preliminary']),
        ('validation', {'errors': validated['errors'], 'warnings': validated['warnings'],
                        'one_to_one_mapping': validated['one_to_one'],
                        'test_records_found': len(validated['test_rows'])}),
        ('derived', {'usable_cough': 'clear_cough OR mixed_noisy_cough',
                     'high_quality_cough': 'clear_cough only'}),
        ('overall', {'counts': overall['counts'], 'pct': overall['pct'],
                     'usable_cough': overall['usable_cough'],
                     'high_quality_cough': overall['high_quality_cough']}),
        ('groups', OrderedDict((g, dict(groups[g])) for g in groups)),
        ('fisher_usable_cough', fisher_usable),
        ('fisher_clear_cough_vs_other', fisher_clear),
        ('phase5_automated_score_vs_manual', ph5),
        ('recommendations', recommendations),
        ('integrity', {'diagnosis_labels_changed': False, 'splits_changed': False,
                       'test_records_touched': False, 'model_results_overwritten': False,
                       'recordings_removed': False, 'model_retrained': False}),
    ])
    return summary


def row_stats(block):
    out = {'n': block['n']}
    for rating in RATINGS:
        out[f'{rating}_count'] = block['counts'][rating]
        out[f'{rating}_percent'] = round(block['pct'][rating], 1)
    out['usable_cough_count'] = block['usable_cough']['count']
    out['usable_cough_percent'] = round(block['usable_cough']['pct'], 1)
    out['high_quality_cough_count'] = block['high_quality_cough']['count']
    out['high_quality_cough_percent'] = round(block['high_quality_cough']['pct'], 1)
    return out


def write_outputs(out_dir, summary, rows, review_path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'review_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')

    over = summary['overall']
    csv_rows = [{'metric': 'total_reviewed', 'group': 'overall', 'count': summary['completed_reviews'],
                 'percent': '100.0', 'note': 'completed of 90'},
                {'metric': 'missing', 'group': 'overall', 'count': summary['missing_count'],
                 'percent': round(pct(summary['missing_count'], REVIEW_COUNT), 1), 'note': 'unreviewed'}]
    for rating in RATINGS:
        csv_rows.append({'metric': rating, 'group': 'overall',
                         'count': over['counts'][rating], 'percent': round(over['pct'][rating], 1),
                         'note': 'manual rating'})
    csv_rows.append({'metric': 'usable_cough', 'group': 'overall',
                     'count': over['usable_cough']['count'],
                     'percent': round(over['usable_cough']['pct'], 1),
                     'note': 'clear_cough OR mixed_noisy_cough'})
    csv_rows.append({'metric': 'high_quality_cough', 'group': 'overall',
                     'count': over['high_quality_cough']['count'],
                     'percent': round(over['high_quality_cough']['pct'], 1),
                     'note': 'clear_cough only'})
    for g in ('flagged', 'comparison'):
        blk = summary['groups'][g]
        csv_rows.append({'metric': 'total_reviewed', 'group': g, 'count': blk['n'],
                         'percent': '100.0', 'note': 'records in group'})
        for rating in RATINGS:
            csv_rows.append({'metric': rating, 'group': g, 'count': blk['counts'][rating],
                             'percent': round(blk['pct'][rating], 1), 'note': 'manual rating'})
        csv_rows.append({'metric': 'usable_cough', 'group': g,
                         'count': blk['usable_cough']['count'],
                         'percent': round(blk['usable_cough']['pct'], 1), 'note': 'derived'})
        csv_rows.append({'metric': 'high_quality_cough', 'group': g,
                         'count': blk['high_quality_cough']['count'],
                         'percent': round(blk['high_quality_cough']['pct'], 1), 'note': 'derived'})
    with (out_dir / 'review_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'group', 'count', 'percent', 'note'])
        w.writeheader()
        w.writerows(csv_rows)

    with (out_dir / 'flagged_vs_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['group'] + [k for k in row_stats(summary['groups']['flagged']) if k != 'n']
        w = csv.writer(f)
        w.writerow(fields)
        w.writerow(['flagged'] + [row_stats(summary['groups']['flagged'])[k] for k in fields[1:]])
        w.writerow(['comparison'] + [row_stats(summary['groups']['comparison'])[k] for k in fields[1:]])

    with (out_dir / 'review_joined_private.csv').open('w', newline='', encoding='utf-8') as f:
        f.write('# PRIVATE - contains real recording UUIDs from listening-review-key.csv. '
                'Do not copy into index.html or any public artifact.\n')
        w = csv.DictWriter(f, fieldnames=['review_id', 'rating', 'usable_cough',
                                          'high_quality_cough', 'notes', 'uuid', 'split',
                                          'label', 'group', 'cough_detected'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['review_id']):
            w.writerow({'review_id': r['review_id'], 'rating': r['rating'],
                        'usable_cough': int(r['usable_cough']),
                        'high_quality_cough': int(r['high_quality_cough']),
                        'notes': r['notes'], 'uuid': r['uuid'], 'split': r['split'],
                        'label': r['label'], 'group': r['group'],
                        'cough_detected': r.get('cough_detected')})

    (out_dir / 'review_analysis.md').write_text(
        render_markdown(summary), encoding='utf-8')


def fmt_p(v):
    return 'n/a' if v is None else f'{v:.4f}'


def render_markdown(summary):
    lines = []
    lines.append('# Blinded cough-content listening review - analysis')
    lines.append('')
    lines.append(f'Batch: `{summary["batch"]}` (expected `{summary["batch_expected"]}`). '
                 f'Source review file: `{summary["review_source"]}`.')
    lines.append('')
    if summary['preliminary']:
        lines.append('## **PRELIMINARY - review incomplete**')
        lines.append('')
        lines.append(f'**{summary["completed_reviews"]} of {summary["expected_reviews"]} '
                     f'recordings reviewed; {summary["missing_count"]} missing.** '
                     f'Results below summarize completed reviews only and must not be treated as '
                     f'final inference. Rerun after the review is complete or otherwise finalized.')
        lines.append('')
    if summary['validation']['errors']:
        lines.append('## Validation errors')
        lines.append('')
        for e in summary['validation']['errors']:
            lines.append(f'- {e}')
        lines.append('')
    if summary['validation']['warnings']:
        lines.append('## Validation warnings')
        lines.append('')
        for w in summary['validation']['warnings']:
            lines.append(f'- {w}')
        lines.append('')
    lines.append('Definitions: **usable_cough** = clear_cough OR mixed_noisy_cough. '
                 '**high_quality_cough** = clear_cough only. Ratings describe audible content; '
                 'they are not diagnostic labels.')
    lines.append('')
    over = summary['overall']
    lines.append('## Overall')
    lines.append('')
    lines.append(f'- Reviewed: {summary["completed_reviews"]}/{summary["expected_reviews"]}'
                 f' (missing: {summary["missing_count"]})')
    for rating in RATINGS:
        lines.append(f'- {rating}: {over["counts"][rating]} '
                     f'({over["pct"][rating]:.1f}%)')
    lines.append(f'- usable_cough: {over["usable_cough"]["count"]} '
                 f'({over["usable_cough"]["pct"]:.1f}%)')
    lines.append(f'- high_quality_cough: {over["high_quality_cough"]["count"]} '
                 f'({over["high_quality_cough"]["pct"]:.1f}%)')
    lines.append('')
    lines.append('## Flagged vs comparison (exact counts)')
    lines.append('')
    lines.append('| group | N | clear | mixed/noisy | no cough | uncertain | usable | high-quality |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for g in ('flagged', 'comparison'):
        b = summary['groups'][g]
        lines.append(f'| {g} | {b["n"]} '
                     f'| {b["counts"]["clear_cough"]} ({b["pct"]["clear_cough"]:.1f}%) '
                     f'| {b["counts"]["mixed_noisy_cough"]} ({b["pct"]["mixed_noisy_cough"]:.1f}%) '
                     f'| {b["counts"]["no_cough"]} ({b["pct"]["no_cough"]:.1f}%) '
                     f'| {b["counts"]["uncertain"]} ({b["pct"]["uncertain"]:.1f}%) '
                     f'| {b["usable_cough"]["count"]} ({b["usable_cough"]["pct"]:.1f}%) '
                     f'| {b["high_quality_cough"]["count"]} ({b["high_quality_cough"]["pct"]:.1f}%) |')
    lines.append('')
    lines.append('Percentages cannot be called better or worse by themselves; the group difference '
                 'is tested with Fisher below.')
    lines.append('')
    fu = summary['fisher_usable_cough']
    fc = summary['fisher_clear_cough_vs_other']
    lines.append('## Statistical comparison (Fisher exact)')
    lines.append('')
    lines.append('### Usable cough (clear OR mixed/noisy) in flagged vs comparison')
    lines.append('')
    lines.append(f'- 2x2 table {fu["table"]} ({fu["definition"]})')
    lines.append(f'- Odds ratio (flagged relative to comparison): {fmt_p(fu["oddsratio"])}')
    if fu['ci95']:
        lines.append(f'- 95% confidence interval: [{fu["ci95"][0]:.4f}, {fu["ci95"][1]:.4f}]')
    lines.append(f'- Fisher exact p-value (two-sided): {fmt_p(fu["p_value"])}')
    if fu['note']:
        lines.append(f'- Note: {fu["note"]}')
    lines.append('')
    lines.append('### Clear cough only vs all other ratings')
    lines.append('')
    lines.append(f'- 2x2 table {fc["table"]} ({fc["definition"]})')
    lines.append(f'- Odds ratio (flagged relative to comparison): {fmt_p(fc["oddsratio"])}')
    if fc['ci95']:
        lines.append(f'- 95% confidence interval: [{fc["ci95"][0]:.4f}, {fc["ci95"][1]:.4f}]')
    lines.append(f'- Fisher exact p-value (two-sided): {fmt_p(fc["p_value"])}')
    if fc['note']:
        lines.append(f'- Note: {fc["note"]}')
    lines.append('')
    ph5 = summary['phase5_automated_score_vs_manual']
    lines.append('## Relationship to the existing automated cough-presence score')
    lines.append('')
    lines.append(f'- Field: {ph5["field"]} from {ph5["source"]}; development recordings only.')
    lines.append(f'- Reviewed records with a score: {ph5["n_with_score"]}')
    lines.append(f'- Median score, manual usable={ph5["usable_n"]}: {fmt_p(ph5["usable_median"])}')
    lines.append(f'- Median score, manual not-usable={ph5["not_usable_n"]}: {fmt_p(ph5["not_usable_median"])}')
    if ph5['mann_whitney']:
        mw = ph5['mann_whitney']
        lines.append(f'- Mann-Whitney U usable vs not-usable: U={mw["u_statistic"]:.1f}, '
                     f'p={fmt_p(mw["p_value"])}')
    if ph5['spearman']:
        sp = ph5['spearman']
        lines.append(f'- Spearman rho (score vs usable_cough): {fmt_p(sp["rho"])}, '
                     f'p={fmt_p(sp["p_value"])}')
    lines.append('')
    lines.append('**Caveat:** ' + ph5['caveat'])
    lines.append('')
    lines.append('## Recommendations (evidence-based next-step options)')
    lines.append('')
    for b in summary['recommendations']:
        lines.append(f'- {b}')
    lines.append('')
    lines.append('## Boundaries')
    lines.append('')
    lines.append('These ratings are manual judgments of audible cough content, not URTI '
                 'diagnoses. They were never used to relabel diagnosis, change splits, touch '
                 'test records, retrain models, or overwrite V1/V2/V3/V4 outputs.')
    lines.append('')
    lines.append('The private joined file `review_joined_private.csv` contains real recording '
                 'UUIDs and must not be copied into public artifacts.')
    return '\n'.join(lines)


def load_dev_scores(labels_path):
    scores = {}
    if labels_path and labels_path.exists():
        for row in csv.DictReader(labels_path.open(encoding='utf-8-sig')):
            uuid = row.get('uuid')
            try:
                scores[uuid] = float(row['cough_detected'])
            except (TypeError, ValueError, KeyError):
                scores[uuid] = None
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser(description='Analyze the blinded listening-review JSON.')
    ap.add_argument('--review', type=Path,
                    default=Path(__file__).resolve().parent / 'urti-audio-reviews.json')
    ap.add_argument('--key', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'listening-review-key.csv')
    ap.add_argument('--dev-labels', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'development-audit' / 'labels.csv')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).resolve().parent / 'analysis')
    args = ap.parse_args(argv)

    if not args.review.exists():
        print(f'Review file not found: {args.review}')
        print()
        print('Place the browser export here:')
        print(f'    {args.review}')
        print()
        print('When it exists, run:')
        print(f'    python {Path(__file__).name} --review "{args.review}"')
        return 2
    try:
        data = json.loads(args.review.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f'Could not parse review file {args.review}: {exc}')
        return 2
    key_rows = load_key(args.key)
    validated = validate(data, key_rows)
    rows = attach_ratings(validated)
    dev_scores = load_dev_scores(args.dev_labels)
    for r in rows:
        r['cough_detected'] = dev_scores.get(r['uuid'])
    if validated['errors']:
        print('Validation issues:')
        for e in validated['errors']:
            print('  ERROR: ' + e)
    for w in validated['warnings']:
        print('  WARN: ' + w)
    if validated['errors']:
        print('Aborting before producing outputs; fix the reported issues first.')
        return 1
    summary = build_summary(rows, validated, args.key, args.review)
    write_outputs(args.out, summary, rows, args.review)
    print(f'Wrote analysis outputs to: {args.out}')
    print(f'  completed reviews: {summary["completed_reviews"]}/{summary["expected_reviews"]}')
    if summary['preliminary']:
        print('  WARNING: analysis is PRELIMINARY (missing entries present).')
    print('  Fisher usable-cough p = ' + fmt_p(summary['fisher_usable_cough']['p_value']))
    return 0


if __name__ == '__main__':
    sys.exit(main())