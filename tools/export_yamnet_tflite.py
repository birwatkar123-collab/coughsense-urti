"""Canonical, reproducible YAMNet -> TFLite conversion for the Android app.

Runs in D:\\urti\\.venv-tf (tensorflow-hub, tensorflow 2.16.2). Downloads/uses the cached
official tfhub module, wraps it as a dynamic-length float32 waveform function returning
the frame embeddings and their mean, converts to TFLite (BUILTINS only), and verifies
the artifact reproduces the original module's mean embedding to <1e-4 on a reference signal.

Output: model-artifacts/yamnet_dynamic.tflite  (byte-identical rebuild of the shipped asset)
"""
import io
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

HUB_URL = 'https://tfhub.dev/google/yamnet/1'
OUT = None


def make_wrapped(hub_model):
    @tf.function(input_signature=[tf.TensorSpec([1, None], tf.float32, name='waveform')])
    def yamnet(waveform):
        scores, embeddings, _ = hub_model(waveform)
        return {'embeddings': embeddings, 'mean': tf.reduce_mean(embeddings, axis=0)}
    return yamnet


def main():
    global OUT
    import pathlib
    OUT = pathlib.Path(__file__).resolve().parents[1] / 'model-artifacts'

    m = hub.load(HUB_URL)
    wrapped = make_wrapped(m)
    concrete = wrapped.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()

    from tensorflow.lite.python.interpreter import Interpreter

    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    n_in = interp.get_input_details()[0]['index']
    n_emb = [d['index'] for d in interp.get_output_details() if d['name'].startswith('Identity_')][0]
    n_mean = [d['index'] for d in interp.get_output_details() if d['name'] == 'Identity_1' or (d['index'] != n_emb and d['shape'].size == 1)][0]

    rng = np.random.default_rng(7)
    test = (rng.normal(0, 0.1, 87_000)).astype(np.float32)
    interp.resize_tensor_input(n_in, [len(test)], strict=False)
    interp.allocate_tensors()
    interp.set_tensor(n_in, test)
    interp.invoke()

    lite_mean = interp.get_tensor(n_mean).astype(np.float64)
    tf_mean = np.asarray(wrapped(test)['mean']).astype(np.float64)
    max_diff = float(np.max(np.abs(tf_mean - lite_mean)))
    print(f'reference max-abs-diff (tfhub vs tflite mean embedding): {max_diff:.3e}')
    if max_diff >= 1e-4:
        sys.exit('CONVERSION MISMATCH')

    out_file = OUT / 'yamnet_dynamic.tflite'
    out_file.write_bytes(tflite_bytes)
    print(f'wrote {out_file} ({len(tflite_bytes)} bytes)')


if __name__ == '__main__':
    main()