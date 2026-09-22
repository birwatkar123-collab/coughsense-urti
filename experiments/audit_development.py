"""Metadata audit plus stratified audio-quality spot check; excludes test data."""
import csv, io, json, random, subprocess, sys, tempfile, zipfile
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.audio-audit'))
import imageio_ffmpeg
import numpy as np

OUT = ROOT / 'experiments/development-audit'
OUT.mkdir(exist_ok=True)
manifest = list(csv.DictReader((ROOT / 'experiments/v2-source/outputs/manifest.csv').open()))
dev = [r for r in manifest if r['split'] != 'test']
assert len(dev) == 2093
metadata = {r['uuid']: r for r in csv.DictReader((ROOT / 'dataset_metadata_original.csv').open(encoding='utf-8-sig'))}

def summary(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return {'n': len(a), 'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())} if len(a) else {'n': 0}

def num(v):
    try: return float(v)
    except (ValueError, TypeError): return float('nan')

by_class = {}
annotators = defaultdict(Counter)
detail = []
for label in (0, 1):
    subset = [r for r in dev if int(r['label']) == label]
    counts, diagnoses, quality, scores, slot_counts = Counter(), Counter(), Counter(), [], Counter()
    for row in subset:
        m = metadata[row['uuid']]
        slots = [i for i in range(1,5) if m[f'diagnosis_{i}']]
        votes = [m[f'diagnosis_{i}'] for i in slots]
        assert votes and {int(v=='upper_infection') for v in votes} == {label}
        assert not any(m[f'quality_{i}'] in ('poor','no_cough') for i in range(1,5))
        counts[len(slots)] += 1
        score = num(m['cough_detected']); scores.append(score)
        for i in slots:
            diagnoses[m[f'diagnosis_{i}']] += 1
            quality[m[f'quality_{i}'] or 'missing'] += 1
            slot_counts[i] += 1
            annotators[i][label] += 1
        detail.append({'uuid':row['uuid'],'split':row['split'],'label':label,
            'expert_count':len(slots),'annotation_slots':','.join(map(str,slots)),
            'diagnoses':','.join(votes),'cough_detected':score})
    by_class[label] = {'recordings':len(subset),'expert_count_distribution':dict(counts),
        'diagnosis_votes':dict(diagnoses),'quality_votes':dict(quality),
        'annotation_slots':dict(slot_counts),'cough_detected_summary':summary(scores),
        'cough_detected_below_0_8':sum(s < .8 for s in scores),
        'cough_detected_below_0_5':sum(s < .5 for s in scores)}

# Random fixed sample: 12 recordings per split/class. Signal checks do not identify coughs.
rng = random.Random(75)
sample = []
for split in ('train','validation'):
    for label in ('0','1'):
        eligible = sorted([r for r in dev if r['split']==split and r['label']==label],key=lambda r:r['uuid'])
        sample.extend(rng.sample(eligible,12))
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
audio_results = []
with zipfile.ZipFile(ROOT / 'kaggle/urti_kaggle_dataset.zip') as z, tempfile.TemporaryDirectory(prefix='urti-audit-') as td:
    for index, r in enumerate(sample):
        path = Path(td) / Path(r['path']).name
        path.write_bytes(z.read(r['path']))
        result = dict(uuid=r['uuid'], split=r['split'],label=int(r['label']))
        try:
            decoded = subprocess.run([ffmpeg,'-v','error','-i',str(path),'-vn','-ac','1','-ar','16000','-t','61','-f','f32le','pipe:1'],capture_output=True,check=True,timeout=60)
            y = np.frombuffer(decoded.stdout,dtype='<f4')
            assert len(y)>0 and np.isfinite(y).all()
            peak = float(np.max(np.abs(y)))
            n = len(y)//320
            rms = np.sqrt(np.mean(y[:n*320].reshape(n,320)**2,axis=1))
            active = rms > max(1e-5, float(rms.max())*.0316228)
            where = np.flatnonzero(active)
            result.update(duration_seconds=len(y)/16000,peak=peak,
                clipped_sample_fraction=float(np.mean(np.abs(y)>=.999)),
                active_frame_fraction=float(active.mean()),
                active_span_seconds=float((where[-1]-where[0]+1)*.02) if len(where) else 0,
                status='decoded')
        except Exception as exc:
            result.update(status='error',error=type(exc).__name__+': '+str(exc)[:200])
        audio_results.append(result)
        path.unlink()
        if (index+1)%12==0: print(f'Audio spot check {index+1}/48',flush=True)

report = {'development_recordings':len(dev),'test_inspected':False,'by_class':by_class,
    'annotation_slot_counts':{str(k):dict(v) for k,v in annotators.items()},
    'audio_sample_size':len(audio_results),'audio_decoded':sum(r['status']=='decoded' for r in audio_results),
    'audio_duration':summary([r['duration_seconds'] for r in audio_results if r['status']=='decoded']),
    'audio_with_over_1_percent_clipped_samples':sum(r.get('clipped_sample_fraction',0)>.01 for r in audio_results),
    'audio_active_span_below_0_2_seconds':sum(r.get('active_span_seconds',999)<.2 for r in audio_results),
    'limitations':['Energy checks do not distinguish cough, speech or background noise.',
        'Random 48-file sample does not certify the full cohort.',
        'Annotation slot indices are not independently verified physician identifiers.',
        'cough_detected is an existing model score, not manual verification.']}
(OUT/'results.json').write_text(json.dumps(report,indent=2))
for name, rows in [('labels.csv',detail),('audio_spot_check.csv',audio_results)]:
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps(report,indent=2))
