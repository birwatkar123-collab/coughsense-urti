"""Write V4 classifier JVM test fixtures (dev validation rows with archived scores).

Run in D:\\urti\\.venv-tf. Selects 3 development/validation uuids that have archived
raw scores, copies their V2 mean embeddings + expected probability to
app/src/test/resources/gold/v4_fixtures.json for the Kotlin unit test.
No test-recording data.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / 'experiments/v2-source/outputs/yamnet_embeddings.npy'
MAN = ROOT / 'experiments/v2-source/outputs/manifest.csv'
PRED = ROOT / 'experiments/v4-validation/validation_predictions.csv'
WEIGHTS = ROOT / 'model-artifacts/v4_classifier/v4_weights.json'
OUT = ROOT / 'app/src/test/resources/gold/v4_fixtures.json'


def main():
    man = list(csv.DictReader(MAN.open(newline='')))
    uuid_idx = {r['uuid']: i for i, r in enumerate(man)}
    preds = {r['uuid']: r for r in csv.DictReader(PRED.open(newline=''))}
    emb = np.load(EMB)

    rng = random.Random(42)
    uuids = rng.sample(sorted(preds), 3)

    w = json.loads(WEIGHTS.open(encoding='utf-8').read())
    units = w['logistic']['coef']
    mean = w['scaler']['mean']
    scale = w['scaler']['scale']
    bron = w['logistic']['intercept'][0]

    fixtures = []
    for u in uuids:
        x = np.asarray(emb[uuid_idx[u]], dtype=np.float64)
        z = float(np.dot((x - np.array(mean)) / np.array(scale), np.array(units)) + bron)
        p = 1.0 / (1.0 + np.exp(-z))
        assert abs(p - float(preds[u]['raw_score'])) < 1e-6, (u, p, preds[u]['raw_score'])
        fixtures.append({'uuid': u, 'embedding': x.astype(object).tolist(),
                         'expected_raw_score': p, 'archived_raw_score': float(preds[u]['raw_score'])})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'positive_class': 1, 'fixtures': fixtures}, indent=2, sort_keys=False), encoding='utf-8')
    print(f'wrote {OUT} ({OUT.stat().st_size} bytes, {len(fixtures)} fixtures)')


if __name__ == '__main__':
    main()