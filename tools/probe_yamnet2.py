"""Probe 2: dynamic-None conversion + fixed-length leading-K truncation equivalence."""
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')
HUB = 'https://tfhub.dev/google/yamnet/1'
OUT = Path(__file__).resolve().parents[1] / 'model-artifacts'
MAXL = 960000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    yamnet = hub.load(HUB)

    def frame_count(L):
        _, emb, _ = yamnet(tf.zeros([L], tf.float32))
        return int(emb.shape[0])

    print('frame counts:', {L: frame_count(L) for L in [9600, 15360, 16000, 32000, 48000, 100000, 480000, 960000]}, flush=True)

    # ---- dynamic None conversion (correct API usage) ----
    mod = tf.Module()
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name='waveform')])
    def mean_embed(waveform):
        _, emb, _ = yamnet(waveform)
        return {'embeddings': emb, 'mean': tf.reduce_mean(emb, axis=0)}
    mod.mean_embed = mean_embed
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([mean_embed.get_concrete_function()], mod)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        blob = converter.convert()
        (OUT / 'yamnet_dynamic.tflite').write_bytes(blob)
        interp = tf.lite.Interpreter(model_content=blob)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        print(f'dynamic conversion OK bytes={len(blob)} input_shape={inp["shape"]} name={inp["name"]}', flush=True)
    except Exception as exc:
        print(f'dynamic conversion FAILED: {type(exc).__name__}: {exc}', flush=True)

    # ---- fixed-length equivalence test ----
    fixed = tf.lite.Interpreter(model_path=str(OUT / 'probe_fixed_960000.tflite'))
    fixed.allocate_tensors()
    din = fixed.get_input_details()[0]
    douts = fixed.get_output_details()
    out_by_name = {o['name'].split(':')[0]: o for o in douts}

    rng = np.random.default_rng(7)
    for trial, L in enumerate([15360, 30000, 120000, 955555]):
        if L < 15600:
            base = np.zeros(L, np.float32); base[4000:6000] = 0.5
        else:
            base = rng.normal(0, 0.15, L).astype(np.float32)
            base[L // 2: L // 2 + 6000] += 0.5
        base = np.clip(base, -1, 1).astype(np.float32)
        _, emb_ref, _ = yamnet(base)
        ref_mean = emb_ref.numpy().mean(axis=0)
        K = emb_ref.shape[0]

        padded = np.zeros(MAXL, np.float32); padded[:len(base)] = base
        if len(din['shape']) == 2:
            fixed.set_tensor(din['index'], padded[np.newaxis, :])
        else:
            fixed.set_tensor(din['index'], padded)
        fixed.invoke()
        # find output index for embeddings
        emb_k = None
        for o in douts:
            shp = o['shape']
            if len(shp) == 2 and shp[1] == 1024:
                emb_k = fixed.get_tensor(o['index'])
                break
        trunc_mean = emb_k[:K].astype(np.float32).mean(axis=0)
        diff = float(np.max(np.abs(ref_mean - trunc_mean)))
        cos = float(np.dot(ref_mean, trunc_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(trunc_mean) + 1e-12))
        print(f'trial L={L:7d} K={K:4d} ref_frames={emb_ref.shape[0]:4d} max|diff|={diff:.3e} cos={cos:.8f}', flush=True)

    # ---- dynamic runtime resize check ----
    try:
        dyn = tf.lite.Interpreter(model_path=str(OUT / 'yamnet_dynamic.tflite'))
        dyn.resize_tensor_input(0, [30000]); dyn.allocate_tensors()
        w = rng.normal(0, 0.1, 30000).astype(np.float32)
        dyn.set_tensor(0, w)
        dyn.invoke()
        od = dyn.get_tensor(dyn.get_output_details()[0]['index'])
        _, emb, _ = yamnet(w)
        print('dynamic resize run ok, out shape', od.shape, 'ref frames', emb.shape[0],
              'maxdiff', float(np.max(np.abs(od - emb.numpy()))), flush=True)
    except Exception as exc:
        print(f'dynamic resize FAILED: {type(exc).__name__}: {exc}', flush=True)


if __name__ == '__main__':
    main()