"""Probe 3: inspect actual tflite input signatures + runtime length behavior."""
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.lite as lite

tf.get_logger().setLevel('ERROR')
OUT = Path(__file__).resolve().parents[1] / 'model-artifacts'


def describe(path, tag):
    print(f'--- {tag} ---', flush=True)
    it = lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    print('input:', inp['name'], 'shape', inp['shape'], 'shape_signature', inp.get('shape_signature'), 'dtype', inp['dtype'])
    outs = it.get_output_details()
    for o in outs:
        print(' output:', o['name'], o['shape'], o.get('shape_signature'))
    it2 = lite.Interpreter(model_path=str(path))
    it2.allocate_tensors()
    # try several lengths by resize + set
    for L in [15360, 30000, 48000, 120000]:
        try:
            it2.resize_tensor_input(0, [L])
            it2.allocate_tensors()
            w = np.zeros([L], np.float32)
            it2.set_tensor(0, w)
            it2.invoke()
            od = it2.get_output_details()
            shapes = {o['name']: it2.get_tensor(o['index']).shape for o in od}
            print(f'  resize [{L}] ok -> {shapes}', flush=True)
        except Exception as exc:
            print(f'  resize [{L}] FAILED: {type(exc).__name__}: {exc}', flush=True)
            return


def main():
    describe(OUT / 'yamnet_dynamic.tflite', 'dynamic')
    describe(OUT / 'probe_fixed_960000.tflite', 'fixed_960000')

    # Forward-verify dynamic model matches tfhub reference on a mid-length signal.
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    rng = np.random.default_rng(3)
    L = 87000
    w = np.clip(rng.normal(0, .1, L).astype(np.float32) + np.where(
        (np.arange(L) % 30000) < 2000, .5, 0.), -1, 1).astype(np.float32)
    _, emb_ref, _ = yamnet(w)
    ref = emb_ref.numpy()

    it = lite.Interpreter(model_path=str(OUT / 'yamnet_dynamic.tflite'))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    if (inp['shape_signature'] is not None and inp['shape_signature'][0] in (-1, 1)):
        it.resize_tensor_input(0, [L])
        it.allocate_tensors()
    elif len(inp['shape']) > 1:
        raise SystemExit('unexpected rank')
    it.set_tensor(0, w)
    it.invoke()
    od = it.get_output_details()
    emb_m = None
    for o in od:
        if len(o['shape']) == 2 and o['shape'][1] == 1024:
            emb_m = it.get_tensor(o['index'])
            break
    print('dynamic model frames', emb_m.shape, 'ref', ref.shape,
          'max|diff|', float(np.max(np.abs(emb_m - ref))), flush=True)


if __name__ == '__main__':
    main()