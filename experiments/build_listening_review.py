import csv, json, random, subprocess, sys, tempfile, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.audio-audit'))
import imageio_ffmpeg
OUT=ROOT/'experiments/listening-review'
(OUT/'audio').mkdir(parents=True,exist_ok=True)
manifest={r['uuid']:r for r in csv.DictReader((ROOT/'experiments/v2-source/outputs/manifest.csv').open())}
flagged=list(csv.DictReader((ROOT/'experiments/development-audit/review_queue.csv').open()))
flag_ids={r['uuid'] for r in flagged}
dev=list(csv.DictReader((ROOT/'experiments/development-audit/labels.csv').open()))
rng=random.Random(20260921)
comparison=rng.sample(sorted([r for r in dev if r['uuid'] not in flag_ids],key=lambda r:r['uuid']),20)
queue=[(r,'flagged') for r in flagged]+[(r,'comparison') for r in comparison]
rng.shuffle(queue)
key=[]
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
with zipfile.ZipFile(ROOT/'kaggle/urti_kaggle_dataset.zip') as z, tempfile.TemporaryDirectory(prefix='urti-review-') as td:
    for n,(r,group) in enumerate(queue,1):
        assert r['split']!='test'
        ident=f'R{n:03d}'
        source=manifest[r['uuid']]['path']
        temp=Path(td)/Path(source).name; temp.write_bytes(z.read(source))
        subprocess.run([ffmpeg,'-v','error','-y','-i',str(temp),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(OUT/'audio'/f'{ident}.wav')],check=True,capture_output=True,timeout=60)
        key.append({'review_id':ident,'uuid':r['uuid'],'split':r['split'],'label':r['label'],'group':group})
        temp.unlink()
        if n%30==0: print(f'Prepared {n}/90',flush=True)
with (OUT.parent/'listening-review-key.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=key[0].keys());w.writeheader();w.writerows(key)
print('Prepared 90 blinded recordings. Mapping stored outside the listening page folder.')
