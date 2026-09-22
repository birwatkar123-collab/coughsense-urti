"""Export V4 classifier weights from the research joblib into app-friendly JSON.

Run with the Python 3.12 interpreter and PYTHONPATH=D:\\urti\\.ml-cpu (sklearn 1.9.1):

    $env:PYTHONPATH='D:\\urti\\.ml-cpu'
    & 'C:\\Users\\mansi\\AppData\\Local\\Python\\pythoncore-3.12-64\\python.exe' tools\\export_v4_weights.py

No retraining, no test data, no scoring changes. Only serializes the already-fitted
raw_model (StandardScaler + LogisticRegression C=0.001 liblinear, balanced) weights,
then verifies end-to-end that the exported weights reproduce the archived raw scores.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / 'experiments/v4-validation'
OUT = ROOT / 'model-artifacts/v4_classifier'


def verify(coef, intercept, mean, scale, m):
    """Reproduce raw_score for every validation row; return per-row abs diffs."""
    features = m.get('features') or []
    if not features:
        man = list(csv.DictReader((ROOT / 'experiments/v2-source/outputs/manifest.csv').open(newline='')))
        uuid_to_idx = {r['uuid']: i for i, r in enumerate(man)}
        emb_mat = np.load(ROOT / 'experiments/v2-source/outputs/yamnet_embeddings.npy')
        feats_of = lambda u: np.asarray(emb_mat[int(uuid_to_idx[u])], dtype=np.float64)
    else:
        man = list(csv.DictReader((ROOT / 'experiments/v2-source/outputs/manifest.csv').open(newline='')))
        uuid_to_idx = {r['uuid']: i for i, r in enumerate(man)}
        emb_mat = np.load(ROOT / 'experiments/v2-source/outputs/yamnet_embeddings.npy')
        feats_of = lambda u: np.asarray(emb_mat[int(uuid_to_idx[u])], dtype=np.float64)
    diffs = []
    with (V4 / 'validation_predictions.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            x = feats_of(row['uuid'])
            z = float(np.dot((x - mean) / scale, coef) + intercept)
            p = 1.0 / (1.0 + np.exp(-z))
            diffs.append(abs(p - float(row['raw_score'])))
    return diffs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blob = V4 / 'v4_models.joblib'
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()

    m = joblib.load(blob)
    pipe = m['raw_model']
    scaler = pipe.named_steps['standardscaler']
    clf = pipe.named_steps['logisticregression']

    coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(clf.intercept_, dtype=np.float64).reshape(-1)[0])
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)

    diffs = verify(coef, intercept, mean, scale, m)
    max_diff = max(diffs) if diffs else float('nan')
    print(f'end-to-end raw_score recompute vs CSV: {len(diffs)} rows, max|diff|={max_diff:.3e}')

    dom = np.arange(clf.n_features_in_)
    payload = {
        'schema_version': 1,
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'joblib_sha256': digest,
        'source': 'experiments/v4-validation/v4_models.joblib (raw_model)',
        'sklearn_version': m.get('sklearn_version'),
        'model_family': 'YAMNet',
        'embedding_pooling': 'mean',
        'classifier_version': 'V4',
        'research_only': True,
        'feature_count': int(clf.n_features_in_),
        'classes': [int(c) for c in clf.classes_.tolist()],
        'positive_class': 1,
        'decision_threshold': 0.5,
        'logistic': {
            'C': clf.C,
            'penalty': clf.penalty,
            'solver': clf.solver,
            'class_weight': clf.class_weight,
            'intercept': [intercept],
            'coef': coef.tolist(),
        },
        'scaler': {
            'with_mean': bool(scaler.with_mean),
            'with_std': bool(scaler.with_std),
            'dims': dom.tolist(),
            'mean': mean.tolist(),
            'scale': scale.tolist(),
        },
        'verification': {'rows_checked': len(diffs), 'max_abs_diff': max_diff},
    }
    out_file = OUT / 'v4_weights.json'
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding='utf-8')
    print(f'wrote {out_file} ({out_file.stat().st_size} bytes)')


if __name__ == '__main__':
    main()