"""Lightweight validation suite for analyze_reviews.py.

Run directly:  python test_analyze_reviews.py
Or under pytest: pytest test_analyze_reviews.py

Covers: fully completed file, empty file, duplicate review IDs, missing IDs,
invalid rating, unknown ID, mapping row mismatch, accidental test-record
inclusion, and batch mismatch. Uses temp dirs so no real outputs are touched.
Reviews used here are synthetic fixtures for the pipeline, clearly not the real
listening result.
"""

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_reviews as ar  # noqa: E402

KEY_PATH = SCRIPT_DIR.parent / 'listening-review-key.csv'
DEV_LABELS = SCRIPT_DIR.parent / 'development-audit' / 'labels.csv'
FIVE_OUTPUTS = {'review_summary.json', 'review_summary.csv',
                'flagged_vs_comparison.csv', 'review_joined_private.csv',
                'review_analysis.md'}


def key_rows():
    return ar.load_key(KEY_PATH)


def full_ratings():
    ratings = ('clear_cough', 'mixed_noisy_cough', 'no_cough', 'uncertain')
    return {'R%03d' % (i + 1): {'rating': ratings[i % 4], 'notes': 'synthetic fixture'}
            for i in range(ar.REVIEW_COUNT)}


def review_data(reviews, batch=ar.EXPECTED_BATCH):
    return {'batch': batch, 'exported_at': '2026-09-21T00:00:00Z', 'reviews': reviews}


def run_pipeline(reviews, out_dir):
    json_path = out_dir / 'urti-audio-reviews.json'
    json_path.write_text(json.dumps(review_data(reviews)), encoding='utf-8')
    return ar.main(['--review', str(json_path), '--key', str(KEY_PATH),
                    '--dev-labels', str(DEV_LABELS), '--out', str(out_dir)])


def test_fully_completed():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rc = run_pipeline(full_ratings(), out)
        assert rc == 0, 'fully completed review should exit 0'
        present = {p.name for p in out.iterdir()}
        assert FIVE_OUTPUTS <= present, f'missing outputs: {FIVE_OUTPUTS - present}'
        summary = json.loads((out / 'review_summary.json').read_text(encoding='utf-8'))
        assert summary['completed_reviews'] == 90
        assert summary['preliminary'] is False
        assert summary['validation']['errors'] == []
        assert summary['fisher_usable_cough']['p_value'] is not None
        private = (out / 'review_joined_private.csv').read_text(encoding='utf-8')
        assert 'PRIVATE' in private.splitlines()[0]
        assert 'uuid' in private


def test_empty_review_file():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rc = run_pipeline({}, out)
        assert rc == 0, 'empty review should summarize (exit 0)'
        summary = json.loads((out / 'review_summary.json').read_text(encoding='utf-8'))
        assert summary['completed_reviews'] == 0
        assert summary['missing_count'] == 90
        assert summary['preliminary'] is True
        assert summary['validation']['errors'] == []
        assert summary['fisher_usable_cough']['p_value'] is None


def test_duplicate_review_ids():
    base = full_ratings()
    as_list = [{'id': i, 'rating': v['rating'], 'notes': v['notes']}
               for i, v in base.items()]
    as_list.append({'id': 'R001', 'rating': 'no_cough', 'notes': 'dup'})
    v = ar.validate(review_data(as_list), key_rows())
    assert any('Duplicate review IDs' in e for e in v['errors'])


def test_missing_ids():
    partial = dict(full_ratings())
    partial.pop('R001')
    partial.pop('R090')
    v = ar.validate(review_data(partial), key_rows())
    assert 'R001' in v['missing'] and 'R090' in v['missing']
    assert v['review_count'] == 88
    assert v['preliminary'] is True
    assert v['errors'] == []


def test_invalid_rating():
    bad = dict(full_ratings())
    bad['R042'] = {'rating': 'maybe', 'notes': ''}
    v = ar.validate(review_data(bad), key_rows())
    assert any('Invalid rating values' in e for e in v['errors'])
    assert any('R042' in e for e in v['errors'])


def test_unknown_review_id():
    unknown = dict(full_ratings())
    unknown['R999'] = {'rating': 'clear_cough', 'notes': ''}
    v = ar.validate(review_data(unknown), key_rows())
    assert any('Unknown review IDs' in e for e in v['errors'])


def test_mapping_row_mismatch():
    rows = key_rows()
    for row in rows:
        if row['review_id'] == 'R090':
            row['review_id'] = 'R999'
    v = ar.validate(review_data(full_ratings()), rows)
    assert any('Mapping row mismatch' in e for e in v['errors'])

    rows2 = key_rows()
    rows2.append(dict(rows2[0]))
    v2 = ar.validate(review_data(full_ratings()), rows2)
    assert any('Mapping duplicate review IDs' in e for e in v2['errors'])


def test_test_record_inclusion():
    rows = key_rows()
    for row in rows:
        if row['review_id'] == 'R001':
            row['split'] = 'test'
    v = ar.validate(review_data(full_ratings()), rows)
    assert any('Test-set records present' in e for e in v['errors'])
    assert v['test_rows'] == ['R001']


def test_batch_mismatch():
    v = ar.validate(review_data(full_ratings(), batch='wrong-batch'), key_rows())
    assert any('Batch mismatch' in e for e in v['errors'])
    assert v['batch_ok'] is False


def run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f'PASS {t.__name__}')
        except AssertionError as exc:
            failures += 1
            print(f'FAIL {t.__name__}: {exc}')
    print(f'\n{len(tests) - failures}/{len(tests)} tests passed '
          f'(synthetic fixtures only; no real review JSON used).')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(run_all())