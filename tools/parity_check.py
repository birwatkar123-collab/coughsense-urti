"""End-to-end parity: on-device-equivalent pipeline vs archived research outputs.

For DEVELOPMENT (train/validation) recordings only. Each row:
  decoded audio (ffmpeg mono 16k f32le) -> research preprocessing (trim/clip/pad)
  -> TFLite YAMNet dynamic model -> mean-pool embedding
  -> compare to archived V2 embedding (yamnet_embeddings.npy)
  -> V4 logistic score (raw_model) -> compare to archived validation raw_score.

Runs in D:\\urti\\.venv-tf (tensorflow 2.16.2 / tflite-runtime / numpy 1.26.4 / imageio-ffmpeg).
"""
import csv
import random
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from app_trim import preprocess_to_embedding_input  # noqa: E402

SR = 16000
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MODEL = ROOT / 'model-artifacts/yamnet_dynamic.tflite'
EMB = ROOT / 'experiments/v2-source/outputs/yamnet_embeddings.npy'
MAN = ROOT / 'experiments/v2-source/outputs/manifest.csv'
PRED = ROOT / 'experiments/v4-validation/validation_predictions.csv'
WEIGHTS = ROOT / 'model-artifacts/v4_classifier/v4_weights.json'


def decode(path):
    r = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-t', '60',
                        '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', 'pipe:1'],
                       capture_output=True, check=True, timeout=180)
    return np.frombuffer(r.stdout, dtype='<f4').copy().astype(np.float32)


def main():
    import json
    interp = Interpreter(model_path=str(MODEL))
    interp.allocate_tensors()
    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    print('input shape', in_details['shape'], 'output', out_details['shape'])

    man = list(csv.DictReader(MAN.open(newline='')))
    uuid_idx = {r['uuid']: i for i, r in enumerate(man)}
    emb = np.load(EMB)

    dev_rows = [r for r in man if r['split'] in ('train', 'validation')]
    rng = random.Random(20260921)
    sample = rng.sample(dev_rows, 12)

    w = json.loads(WEIGHTS.open(encoding='utf-8').read())
    coef = np.array(w['logistic']['coef'], dtype=np.float64)
    bron = w['logistic']['intercept'][0]
    mean = np.array(w['scaler']['mean'], dtype=np.float64)
    scale = np.array(w['scaler']['scale'], dtype=np.float64)

    pred_rows = {r['uuid']: r for r in csv.DictReader(PRED.open(newline=''))}

    checked = 0
    worst_emb = 0.0
    worst_score = 0.0
    for row in sample:
        p = ROOT / 'data/raw/coughvid_20211012' / Path(row.get('path', '')).name
        if not p.exists():
            print('skip missing', row['uuid']); continue
        try:
            y = decode(p)
        except Exception as exc:
            print('skip decode', row['uuid'], exc); continue
        ok, reason, x = preprocess_to_embedding_input(y)
        if not ok:
            print('skip preprocess', row['uuid'], reason); continue

        interp.resize_tensor_input(in_details['index'], [1, len(x)], strict=False)
        interp.allocate_tensors()
        interp.set_tensor(in_details['index'], x.astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(out_details['index'])          # [frames, 1024]
        got = out.mean(axis=0)                                  # mean pooling
        ref = np.asarray(emb[uuid_idx[row['uuid']]], dtype=np.float64)
        emb_diff = float(np.max(np.abs(got.astype(np.float64) - ref)))

        feats = (ref - mean) / scale
        p_ref = 1.0 / (1.0 + np.exp(-(np.dot(feats, coef) + bron)))
        feats2 = (got.astype(np.float64) - mean) / scale
        p_got = 1.0 / (1.0 + np.exp(-(np.dot(feats2, coef) + bron)))
        archived = pred_rows.get(row['uuid'])
        archived_score = float(archived['raw_score']) if archived else None
        worst_emb = max(worst_emb, emb_diff)
        if archived_score is not None:
            worst_score = max(worst_score, abs(p_got - archived_score))
        checked += 1
        tag = 'DEV' if archived_score is not None else 'dev(train, no V4 score)'
        print(f"{row['uuid'][:8]} split={row['split']:10s} dur={len(y)/SR:6.2f}s"
              f" frames={out.shape[0]:4d} emb_maxabs_diff={emb_diff:.3e}"
              f" p_got={p_got:.4f} p_archived={archived_score if archived_score is not None else float('nan'):.4f} {tag}")
        if out.shape[0] != (1 + max(0, (len(x) - 15600 + 7679) // 7680)):
            print('  !! frame-count formula mismatch', len(x), out.shape[0])

    print(f'checked {checked} dev rows; worst embedding max-abs-diff {worst_emb:.3e}; '
          f'worst raw_score diff vs archived {worst_score:.3e}')
    ok_emb = worst_emb < 5e-3 if checked else False
    ok_score = worst_score < 5e-3 if checked else False
    print('PARITY', 'PASS' if (ok_emb and ok_score) else 'FAIL')


if __name__ == '__main__':
    main()