"""One deterministic audio transform shared by training and inference."""
import json
import subprocess
from pathlib import Path
import librosa
import numpy as np


def load_config(path):
    return json.loads(Path(path).read_text())


def transform(path, cfg):
    # ffmpeg handles the dataset's WebM, Ogg and WAV formats consistently.
    result = subprocess.run([
        'ffmpeg', '-v', 'error', '-i', str(Path(path).resolve()),
        '-t', str(cfg['max_input_seconds'] + 1), '-vn', '-ac', '1',
        '-ar', str(cfg['sample_rate']), '-f', 'f32le', 'pipe:1'
    ], capture_output=True, check=True, timeout=90)
    y = np.frombuffer(result.stdout, dtype='<f4').copy()
    sr = cfg['sample_rate']
    if len(y) < sr // 5 or len(y) > sr * cfg['max_input_seconds']:
        raise ValueError('Recording must be between 0.2 and 60 seconds.')
    if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-5:
        raise ValueError('Silent or invalid recording.')
    y, _ = librosa.effects.trim(y, top_db=cfg['trim_top_db'])
    if len(y) < sr // 5:
        raise ValueError('Too little audible content.')
    y = y / max(float(np.max(np.abs(y))), 1e-8)
    length = int(sr * cfg['duration_seconds'])
    # Centre crop after trimming; right-pad shorter recordings with silence.
    start = max(0, (len(y) - length) // 2)
    y = y[start:start + length]
    y = np.pad(y, (0, length - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
        n_mels=cfg['n_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'],
        power=2.0, center=True, pad_mode='constant', htk=False, norm='slaney')
    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return ((db + 80.0) / 80.0).astype(np.float32)[..., None]
