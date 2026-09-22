"""Verify cohort policy, archive integrity, notebook syntax and split isolation."""
import ast
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from prepare_package import select_label

root = Path(__file__).resolve().parent
assert select_label({}) is None
assert select_label({'diagnosis_1': 'upper_infection'}) == 1
assert select_label({'diagnosis_1': 'healthy_cough', 'diagnosis_2': 'lower_infection'}) == 0
assert select_label({'diagnosis_1': 'upper_infection', 'diagnosis_2': 'COVID-19'}) is None
assert select_label({'diagnosis_1': 'upper_infection', 'quality_2': 'no_cough'}) is None
assert select_label({'diagnosis_1': 'upper_infection', 'quality_1': 'poor'}) is None
try:
    select_label({'diagnosis_1': 'unknown_diagnosis'})
except ValueError:
    pass
else:
    raise AssertionError('Unknown diagnoses must not become negative examples')

nb = json.loads((root / 'URTI_Training.ipynb').read_text())
assert nb['nbformat'] == 4
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'notebook cell {i}', 'exec')
for script in root.glob('*.py'):
    ast.parse(script.read_text())
with zipfile.ZipFile(root / 'urti_kaggle_dataset.zip') as z:
    assert z.testzip() is None
    rows = list(csv.DictReader(io.StringIO(z.read('manifest.csv').decode())))
    assert len(rows) == 2465
    assert len({r['uuid'] for r in rows}) == len(rows)
    groups = {}
    for row in rows:
        assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
        groups.setdefault(row['sha256'], set()).add(row['split'])
    assert all(len(splits) == 1 for splits in groups.values())
    for name in ['train.py', 'preprocess.py', 'predict.py', 'preprocessing.json', 'README.md']:
        assert z.read(name) == (root / name).read_bytes()
print('PASS: label-policy cases, Python/notebook syntax, archive CRC, 2465 audio hashes, split isolation, bundled source match.')
