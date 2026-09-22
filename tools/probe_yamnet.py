"""Probe YAMNet tfhub model: signatures, dynamic inputs, TFLite conversion."""
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')
HUB = 'https://tfhub.dev/google/yamnet/1'
CACHE = Path(__file__).resolve().parents[1] / '.yamnet-cache'
OUT = Path(__file__).resolve().parents[1] / 'model-artifacts'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('Loading YAMNet from tfhub...', flush=True)
    yamnet = hub.load(HUB)
    print(f'Loaded in {time.time() - t0:.1f}s', flush=True)

    # Explore the signature.
    if hasattr(yamnet, 'signatures'):
        print('signatures:', list(yamnet.signatures), flush=True)
        for k, v in yamnet.signatures.items():
            print(' ', k, {kk: (i.shape.as_list(), i.dtype.name) for kk, i in v.structured_input_signature[1].items()})
            print('    ->', {kk: (o.shape.as_list(), o.dtype.name) for kk, o in v.structured_outputs.items()})

    wave = np.zeros(16000, dtype=np.float32)
    r = yamnet(wave)
    print('call(waveform) returns:', [type(x).__name__ + str(getattr(x, 'shape', '')) for x in r], flush=True)
    for i, x in enumerate(r):
        a = x.numpy() if hasattr(x, 'numpy') else x
        print(f'  [{i}] shape={getattr(a, "shape", None)} dtype={getattr(a, "dtype", None)}')

    # Try a fixed-ish dynamic wrapper producing mean-pooled embedding.
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name='waveform')])
    def mean_embed(waveform):
        scores, emb, _ = yamnet(waveform)
        mean = tf.reduce_mean(emb, axis=0)
        return {'embeddings': emb, 'mean': mean}

    # Verify shapes on sample inputs.

    precursor = tf.python.module.Module() if False else None
    # tf.python.module is private; use exported module via hub.load's internal.
    # Convert both dynamic (None) and fixed-length wrappers.
    attempts = []
    dynamic = tf.function(mean_embed.get_concrete_function(waveform=tf.TensorSpec([None], tf.float32)))
    attempts.append(('dynamic_None', dynamic, None))
    for length in (16000, 15360, 960000):
        try:
            fixed = mean_embed.get_concrete_function(
                waveform=tf.TensorSpec([length], tf.float32))
            attempts.append((f'fixed_{length}', fixed, None))
        except Exception as exc:  # pragma: no cover
            print(f'  concrete fixed_{length} failed: {exc}')

    for name, fn, _ in attempts:
        try:
            print(f'Converting {name}...', flush=True)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([fn], fn)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            tflite = converter.convert()
            p = OUT / f'probe_{name}.tflite'
            p.write_bytes(tflite)
            # Run it with the reference interpreter using the equivalent saved model.
            try:
                import tensorflow.lite as lite
                interp = lite.Interpreter(model_content=tflite)
                interp.allocate_tensors()
                inp = interp.get_input_details()[0]
                out = interp.get_output_details()
                print(f'  {name}: OK bytes={len(tflite)} input={inp["shape"]} out={[(o["name"], o["shape"]) for o in out]}', flush=True)
            except Exception as e:
                print(f'  {name}: converted but interpreter failed: {type(e).__name__}: {e}')
        except Exception as exc:
            print(f'  {name}: CONVERSION FAILED: {type(exc).__name__}: {exc}', flush=True)


if __name__ == '__main__':
    main()