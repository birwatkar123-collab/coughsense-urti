"""Usage: python predict.py recording.webm --bundle outputs"""
import argparse
import json
import math
from pathlib import Path
import tensorflow as tf
from preprocess import load_config, transform


def _platt_score(raw_score, bundle):
    path = Path(bundle) / 'platt_calibration.json'
    if not path.exists():
        return raw_score
    payload = json.loads(path.read_text())
    p = min(max(float(raw_score), 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    z = payload['coef'] * logit + payload['intercept']
    return 1.0 / (1.0 + math.exp(-z))


def predict(audio, bundle):
    bundle = Path(bundle)
    cfg = load_config(bundle / 'preprocessing.json')
    model = tf.keras.models.load_model(bundle / 'urti_model.keras', compile=False)
    x = transform(audio, cfg)
    if tuple(model.input_shape[1:]) != x.shape:
        raise ValueError('Model and preprocessing shape mismatch.')
    raw_score = float(model.predict(x[None], verbose=0)[0, 0])
    score = _platt_score(raw_score, bundle) if cfg.get('calibration') == 'platt' else raw_score
    return {'urti_score': score, 'raw_score': raw_score, 'label': int(score >= cfg['threshold']),
            'threshold': cfg['threshold'], 'calibrated': cfg['calibrated'],
            'interpretation': 'Experimental cough-label estimate; not a diagnosis.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--bundle', default='outputs')
    args = parser.parse_args()
    print(json.dumps(predict(args.audio, args.bundle), indent=2))
