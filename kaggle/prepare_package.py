"""Standard-library-only dataset packager. No decoding or ML dependencies."""
import argparse
import collections
import csv
import hashlib
import io
import json
import random
import uuid
import zipfile
from pathlib import Path

KNOWN = {'upper_infection', 'lower_infection', 'healthy_cough', 'COVID-19', 'obstructive_disease'}
PREPROCESSING_EXCLUDE_UUIDS = {
    # Fails the packaged transform after trim with "Too little audible content."
    'c9f8ed02-90a1-4a0f-9f2c-d3fd04f3ffbd',
}


def select_label(row):
    votes = [row.get(f'diagnosis_{i}', '').strip() for i in range(1, 5)]
    votes = [v for v in votes if v]
    if set(votes) - KNOWN:
        raise ValueError('Unexpected diagnosis value')
    if not votes:
        return None
    binary = {int(v == 'upper_infection') for v in votes}
    if len(binary) != 1:
        return None
    if any(row.get(f'quality_{i}') in {'poor', 'no_cough'} for i in range(1, 5)):
        return None
    return binary.pop()


def make_notebook():
    cells = []
    def md(text):
        cells.append(dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={}, source=text.splitlines(True)))
    def code(text):
        cells.append(dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))
    md('# Train and calibrate URTI cough models\nAttach the private prepared dataset and enable a GPU. This notebook compares several experimental cough classifiers, selects by validation ROC-AUC, calibrates on validation data only, and reports held-out test metrics. It is not a clinical diagnostic system.')
    code('''import importlib.util, subprocess, sys, shutil
from pathlib import Path
packages = {'numpy': 'numpy', 'pandas': 'pandas', 'librosa': 'librosa>=0.10,<0.12',
            'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib'}
missing = [pkg for mod, pkg in packages.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
if importlib.util.find_spec('tensorflow') is None:
    raise RuntimeError('Use a Kaggle TensorFlow-compatible GPU environment; TensorFlow is missing.')
if shutil.which('ffmpeg') is None:
    raise RuntimeError('FFmpeg is missing. Install FFmpeg in this Kaggle runtime before continuing.')
import tensorflow as tf
print('TensorFlow:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU'), 'Enable a GPU accelerator first.'
''')
    code('''# Set an explicit path here only if auto-discovery finds multiple datasets.
DATA_ROOT = None
if DATA_ROOT is None:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('manifest.csv')
                  if (p.parent / 'preprocessing.json').exists() and (p.parent / 'train.py').exists()]
    assert len(candidates) == 1, f'Expected one prepared dataset, found: {candidates}'
    DATA_ROOT = candidates[0]
DATA_ROOT = Path(DATA_ROOT)
import pandas as pd
manifest = pd.read_csv(DATA_ROOT / 'manifest.csv')
display(manifest.groupby(['split', 'label']).size().unstack(fill_value=0))
assert manifest.uuid.is_unique
assert manifest.groupby('sha256')['split'].nunique().max() == 1
print('Dataset:', DATA_ROOT)
''')
    md('## Train, compare, calibrate, and evaluate\nPreprocessing runs on CPU first. Training uses validation ROC-AUC to choose a checkpoint. Platt and isotonic calibration are fitted on the validation split only; the held-out test split is used once for final metrics. If preprocessing fails, inspect `outputs/preprocessing_failures.json` before proceeding.')
    code('''subprocess.run([sys.executable, '-u', str(DATA_ROOT / 'train.py'),
                '--data', str(DATA_ROOT), '--output', '/kaggle/working/outputs',
                '--epochs', '60'], check=True)
''')
    code('''import json
from IPython.display import Image, display, FileLink
print(json.dumps(json.loads(Path('/kaggle/working/outputs/metrics.json').read_text()), indent=2))
display(Image(filename='/kaggle/working/outputs/evaluation.png'))
display(FileLink('/kaggle/working/urti_model_bundle.zip'))
print('Also available in the notebook output files: urti_model_bundle.zip')
''')
    return dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, default=Path('D:/ECGVisionAI/public_dataset_v3.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    target = root / 'urti_kaggle_dataset.zip'
    rows = []
    with zipfile.ZipFile(args.zip) as source:
        metadata = [n for n in source.namelist() if Path(n).name == 'metadata_compiled.csv']
        assert len(metadata) == 1
        originals = list(csv.DictReader(io.StringIO(source.read(metadata[0]).decode('utf-8-sig'))))
        assert len({r['uuid'] for r in originals}) == len(originals)
        audios = {}
        for name in source.namelist():
            p = Path(name)
            if p.suffix.lower() in {'.wav', '.ogg', '.webm'}:
                if p.stem in audios:
                    raise ValueError('Multiple audio files for one recording ID')
                audios[p.stem] = name
        for row in originals:
            if row['uuid'] in PREPROCESSING_EXCLUDE_UUIDS:
                continue
            label = select_label(row)
            if label is None:
                continue
            name = audios[row['uuid']]
            digest = hashlib.sha256(source.read(name)).hexdigest()
            rows.append({'uuid': row['uuid'], 'path': 'audio/' + Path(name).name,
                         'label': label, 'sha256': digest, 'split': '', '_source': name})
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row['sha256']].append(row)
        for group in grouped.values():
            if len({r['label'] for r in group}) != 1:
                raise ValueError('Byte-identical recordings have conflicting labels; review before packaging')
        rng = random.Random(75)
        for label in (0, 1):
            groups = sorted(k for k, v in grouped.items() if v[0]['label'] == label)
            rng.shuffle(groups)
            train_end = int(len(groups) * .70)
            val_end = train_end + int(len(groups) * .15)
            for i, key in enumerate(groups):
                split = 'train' if i < train_end else 'validation' if i < val_end else 'test'
                for row in grouped[key]:
                    row['split'] = split
        assert all(r['split'] for r in rows)
        counts = collections.Counter((r['split'], r['label']) for r in rows)
        assert all(counts[s, c] > 0 for s in ('train', 'validation', 'test') for c in (0, 1))
        summary = {'recordings': len(rows), 'unique_audio_hashes': len(grouped), 'seed': 75,
                   'preprocessing_excluded_uuids': sorted(PREPROCESSING_EXCLUDE_UUIDS),
                   'counts': {f'{s}_label_{c}': n for (s, c), n in sorted(counts.items())},
                   'policy': 'Consistent available expert binary votes; exclude any poor/no_cough quality.',
                   'source': 'https://zenodo.org/records/7024894',
                   'participant_grouping': 'Unavailable; groups are recording UUIDs and identical audio bytes.'}
        manifest = io.StringIO(newline='')
        writer = csv.DictWriter(manifest, fieldnames=['uuid', 'path', 'label', 'sha256', 'split'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: r['uuid']))
        (root / 'manifest.csv').write_text(manifest.getvalue(), encoding='utf-8')
        (root / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
        (root / 'URTI_Training.ipynb').write_text(json.dumps(make_notebook(), indent=2), encoding='utf-8')
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for row in rows:
                output.writestr(row['path'], source.read(row['_source']))
            for name in ['manifest.csv', 'dataset_summary.json', 'preprocessing.json',
                         'preprocess.py', 'train.py', 'predict.py', 'README.md']:
                output.write(root / name, name)
    print(json.dumps(summary, indent=2))
    print(f'Package: {target} ({target.stat().st_size / 1024**2:.1f} MiB)')


if __name__ == '__main__':
    main()
