"""Generate trim gold cases + an app test-resource bundle (no librosa/sklearn).

Uses the numpy port tools/app_trim.py (validated == librosa.effects.trim semantics) as
the reference. Cases: synthetic battery + decoded development recordings
(train/validation only; never test).

Outputs:
  model-artifacts/gold/gold_trim.bin        binary container (numpy reference)
  app/src/test/resources/gold/gold_trim.bin Kotlin JVM test data (whole container works too)
  model-artifacts/gold/gold_trim_index.json
"""
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_trim import preprocess_to_embedding_input, trim_signal  # noqa: F401 (trim only)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model-artifacts/gold'
APP_RES = ROOT / 'app/app/src/test/resources/gold'
SR = 16000
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def decode(path):
    r = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-t', '60',
                        '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', 'pipe:1'],
                       capture_output=True, check=True, timeout=120)
    return np.frombuffer(r.stdout, dtype='<f4').copy().astype(np.float32)


def synthetic_cases():
    n = SR * 8
    cases = []
    sig = np.zeros(n, np.float32); sig[12000:20000] = 0.3; sig[180000:190000] = 0.25
    cases.append(('synth_tone_bursts', sig))
    body = np.zeros(n, np.float32)
    body[5000:n - 7000] = 0.15
    cases.append(('synth_noise_body', body))
    faint = np.zeros(n, np.float32); faint[4000:17000] = 0.02
    cases.append(('synth_faint', faint))
    cases.append(('synth_silent', np.zeros(n, np.float32)))
    cases.append(('synth_short', np.zeros(1000, np.float32)))
    cases.append(('synth_exact_020', np.full(SR // 5, 0.5, np.float32)))
    long_tone = np.zeros(SR * 20, np.float32)
    long_tone[8000:] = (0.4 * np.sin(2 * np.pi * 300 * np.arange(SR * 20 - 8000) / SR)).astype(np.float32)
    cases.append(('synth_tone20s', long_tone))
    return cases


def write_container(path, entries):
    with path.open('wb') as f:
        f.write(b'URTIGOLD')
        f.write(struct.pack('<i', len(entries)))
        for cid, y, yt, s, e in entries:
            cidb = cid.encode('utf-8')
            f.write(struct.pack('<i', len(cidb))); f.write(cidb)
            f.write(struct.pack('<i', len(y))); f.write(y.astype('<f4').tobytes())
            f.write(struct.pack('<i', len(yt))); f.write(yt.astype('<f4').tobytes())
            f.write(struct.pack('<ii', s, e))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    APP_RES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260921)

    import csv
    man = list(csv.DictReader((ROOT / 'experiments/v2-source/outputs/manifest.csv').open(newline='')))
    dev = [r for r in man if r['split'] in ('train', 'validation')]
    dev.sort(key=lambda r: r['uuid'])
    chosen = rng.sample(dev, 16)

    cases = synthetic_cases()
    decoded = 0
    for r in chosen:
        p = ROOT / 'data/raw/coughvid_20211012' / Path(r.get('path', '')).name
        if not p.exists():
            continue
        try:
            y = decode(p)
        except Exception as exc:
            print(f'skip {r["uuid"]}: {exc}')
            continue
        if len(y) < SR * 0.2:
            continue
        cases.append((f'dev_{r["uuid"]}', y))
        decoded += 1

    entries = []
    for cid, y in cases:
        y = np.asarray(y, np.float32)
        s, t = trim_signal(y)
        entries.append((cid, y, t, s, s + len(t)))

    write_container(OUT / 'gold_trim.bin', entries)
    write_container(APP_RES / 'gold_trim.bin', entries)
    with (OUT / 'gold_trim_index.json').open('w') as f:
        json.dump([{'id': cid, 'input_len': int(len(y)), 'out_len': int(len(yt)),
                    'start': s, 'end': e} for cid, y, yt, s, e in entries], f, indent=2)
    print(f'wrote {len(entries)} entries ({decoded} dev decoded) -> gold_trim.bin (+ Kotlin resources)')


if __name__ == '__main__':
    main()