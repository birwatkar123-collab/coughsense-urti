"""CPU experiment: train-only selection and calibration, validation reporting."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.ml-cpu'))
import csv
import hashlib
import json
import warnings
import numpy as np
import sklearn
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

warnings.filterwarnings('error', category=ConvergenceWarning)
SRC = ROOT / 'experiments/v2-source/outputs'
OUT = ROOT / 'experiments/v4-validation'
SEED = 75


def classifier(c):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, solver='liblinear', class_weight='balanced', max_iter=5000, random_state=SEED))


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
        sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
        balanced_accuracy=float((tp / (tp + fn) + tn / (tn + fp)) / 2),
        brier_score=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], threshold=.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise RuntimeError('Completed experiment exists; preserve it instead of overwriting.')
    rows = list(csv.DictReader((SRC / 'manifest.csv').open(newline='')))
    x = np.load(SRC / 'yamnet_embeddings.npy', allow_pickle=False)
    assert x.shape == (2465, 1024) and np.isfinite(x).all()
    assert len(rows) == len(x) and len({r['uuid'] for r in rows}) == len(rows)
    original = {r['uuid']: r for r in csv.DictReader((ROOT / 'kaggle/manifest.csv').open(newline=''))}
    assert set(original) - {r['uuid'] for r in rows} == {'c9f8ed02-90a1-4a0f-9f2c-d3fd04f3ffbd'}
    groups = {}
    for r in rows:
        assert all(r[k] == original[r['uuid']][k] for k in ['label', 'split', 'sha256'])
        groups.setdefault(r['sha256'], set()).add(r['split'])
    assert all(len(v) == 1 for v in groups.values())
    train = np.array([r['split'] == 'train' for r in rows])
    val = np.array([r['split'] == 'validation' for r in rows])
    labels = np.array([int(r['label']) for r in rows])
    xt, yt, xv, yv = x[train], labels[train], x[val], labels[val]
    assert len(yt) == 1724 and len(yv) == 369
    # Test rows are never used for fitting, tuning, calibration or predictions.
    del x, labels
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(xt, yt))
    grid = []
    for c in [.001, .003, .01, .03, .1, .3, 1.0]:
        aucs = cross_val_score(classifier(c), xt, yt, cv=folds, scoring='roc_auc', n_jobs=1, error_score='raise')
        result = dict(C=c, mean_training_cv_auc=float(aucs.mean()), fold_auc=aucs.tolist())
        grid.append(result)
        print(json.dumps(result), flush=True)
    best = max(grid, key=lambda r: r['mean_training_cv_auc'])
    c = best['C']
    raw = classifier(c).fit(xt, yt)
    # Cross-fitted training predictions fit the sigmoid calibrator; validation is not used.
    calibrated = CalibratedClassifierCV(classifier(c), method='sigmoid', cv=folds, ensemble=False, n_jobs=1)
    calibrated.fit(xt, yt)
    p_raw = raw.predict_proba(xv)[:, 1]
    p_cal = calibrated.predict_proba(xv)[:, 1]
    constant = np.full(len(yv), yt.mean())
    result = dict(experiment='V4 balanced logistic regression with train-only selection and sigmoid calibration',
        train_size=len(yt), validation_size=len(yv), selected_C=c,
        selection='5-fold stratified training CV ROC AUC', grid=grid,
        raw=metrics(yv, p_raw), calibrated=metrics(yv, p_cal), constant_baseline=metrics(yv, constant),
        test_evaluated=False, sklearn_version=sklearn.__version__, seed=SEED,
        limitation='Validation has been used in earlier experiments; results remain exploratory. No participant IDs.')
    # Stratified paired bootstrap conditions on this fitted model; not training/selection uncertainty.
    rng = np.random.default_rng(SEED)
    indices = [np.flatnonzero(yv == k) for k in [0, 1]]
    draws = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(i, len(i), replace=True) for i in indices])
        draws.append([roc_auc_score(yv[idx], p_cal[idx]),
            brier_score_loss(yv[idx], p_cal[idx]) - brier_score_loss(yv[idx], constant[idx])])
    limits = np.quantile(draws, [.025, .975], axis=0)
    result['validation_95_percent_bootstrap'] = {'calibrated_auc': limits[:,0].tolist(),
        'calibrated_minus_constant_brier': limits[:,1].tolist(),
        'scope': 'Conditional on fitted model; does not include model-selection uncertainty.'}
    observed, predicted = calibration_curve(yv, p_cal, n_bins=5, strategy='quantile')
    result['calibration_bins'] = dict(mean_score=predicted.tolist(), observed_fraction=observed.tolist())
    result['source_sha256'] = {n:hashlib.sha256((SRC/n).read_bytes()).hexdigest()
                               for n in ['manifest.csv','yamnet_embeddings.npy','run_config.json']}
    with (OUT / 'validation_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['uuid','label','raw_score','calibrated_score'])
        for r, a, b in zip([r for r in rows if r['split']=='validation'], p_raw, p_cal):
            writer.writerow([r['uuid'],r['label'],a,b])
    joblib.dump({'raw_model':raw, 'calibrated_model':calibrated, 'threshold':.5,
        'features':'V2 mean pooled YAMNet embeddings; not an end-to-end audio model',
        'source_config':json.loads((SRC/'run_config.json').read_text()),
        'sklearn_version':sklearn.__version__}, OUT/'v4_models.joblib')
    loaded = joblib.load(OUT/'v4_models.joblib')
    np.testing.assert_allclose(loaded['calibrated_model'].predict_proba(xv)[:,1], p_cal)
    (OUT/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'grid'}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
