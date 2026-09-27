from __future__ import annotations
import hashlib, json, math, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# Scientific protocol stays v0.1.0 so the audited prepared data remain compatible.
VERSION = '0.1.0'
SOFTWARE_VERSION = '0.2.0-dual-gpu'
LABELS = ['cardiomegaly', 'pleural_effusion']
MODELS = {'resnet18': 'resnet18.a1_in1k',
          'densenet121': 'densenet121.ra_in1k',
          'deit_tiny': 'deit_tiny_patch16_224.fb_in1k'}
SEEDS = [17, 43, 101]
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
DOMAINS = ['nih', 'chexpert', 'vinbig']
PROTOCOL = {
    'version': VERSION, 'labels': LABELS, 'models': MODELS,
    'seeds': SEEDS, 'fractions': FRACTIONS,
    'split_seed': 20260926, 'split_proportions': [0.6, 0.1, 0.1, 0.2],
    'size': 224, 'microbatch': 32, 'effective_batch': 64,
    'epochs': 20, 'min_epochs': 5, 'patience': 4, 'min_delta': 1e-4,
    'backbone_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 1e-4,
    'head_dropout': 0.2, 'warmup_epochs': 2, 'loss': 'unweighted_BCE',
    'training_augmentation': 'affine_5deg_translate_0.02;brightness_contrast_0.1;no_flip_no_crop',
    'primary': 'external_Brier_temperature_minus_raw_at_full_training',
    'bootstrap_replicates': 2000, 'ece_bins': 15,
    'mc_head_samples': 20, 'source_threshold_sensitivity': 0.90,
    'calibration_fractions': [0.1, 0.25, 0.5, 1.0],
    'chexpert_patient_cap': 20000,
    'preparation': 'uint8_full_frame_bilinear_224;exact_pixel_dedup;dhash_review',
}

def json_default(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default))
    tmp.replace(path)

def read_json(path): return json.loads(Path(path).read_text())

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def stable_hash(value, seed=0):
    return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()

def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()

def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def check_storage(root='/kaggle/working', reserve_gb=1.0, max_saved_gb=17.0):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    n = directory_bytes(root)
    if n > max_saved_gb * 1e9:
        raise RuntimeError(f'Saved-output safety limit exceeded: {n/1e9:.2f} GB > {max_saved_gb} GB.')
    if shutil.disk_usage(root).free < reserve_gb * 1e9:
        raise RuntimeError('Insufficient scratch disk. Save this session; do not keep large archives in working.')
    return n

def environment(path):
    info = {'python': sys.version, 'platform': platform.platform(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'protocol_sha256': protocol_hash()}
    try:
        info['pip_freeze'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
    except Exception as e: info['pip_error'] = str(e)
    try:
        info['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
    except Exception: info['nvidia_smi'] = 'No NVIDIA GPU detected.'
    write_json(path, info)
    return info

def find_prepared(domain, search_root='/kaggle/input'):
    hits = list(Path(search_root).rglob(f'prepared_{domain}/COMPLETE.json'))
    if not hits: raise FileNotFoundError(f'Attach the completed 00_Prepare_Data output for {domain}.')
    by_fingerprint = {}
    for p in hits:
        info = read_json(p)
        by_fingerprint.setdefault(info['manifest_sha256'], []).append(p.parent)
    if len(by_fingerprint) != 1:
        raise RuntimeError(f'Different prepared versions of {domain} are attached; remove ambiguous inputs.')
    return sorted(next(iter(by_fingerprint.values())), key=str)[0]

def load_prepared(root):
    root = Path(root)
    info = read_json(root / 'COMPLETE.json')
    manifest = root / 'manifest.csv'
    if sha256_file(manifest) != info['manifest_sha256']:
        raise RuntimeError(f'Manifest checksum mismatch: {root}')
    df = pd.read_csv(manifest, dtype={'patient_id': str, 'image_uid': str,
                                    'original_id': str, 'pixel_sha256': str, 'dhash': str})
    if not df.image_uid.is_unique: raise ValueError('Duplicate image_uid.')
    df['_root'] = str(root)
    return df, info

def assert_patient_disjoint(df):
    source = df[df.domain == 'nih']
    bad = source.groupby('patient_id').split.nunique()
    if (bad > 1).any(): raise AssertionError('NIH patients overlap between partitions.')

def nested_patient_subset(df, fraction, seed):
    if not 0 < fraction <= 1: raise ValueError('Fraction must be in (0,1].')
    ids = sorted(df.patient_id.unique(), key=lambda x: stable_hash(x, seed))
    n = len(ids) if fraction == 1 else max(1, math.ceil(len(ids) * fraction))
    return df[df.patient_id.isin(ids[:n])].copy()

def job_id(model, fraction, seed): return f'{model}__f{round(fraction*100):03d}__s{seed}'

def jobs():
    return [dict(model=m, fraction=f, seed=s, job_id=job_id(m, f, s))
            for m in MODELS for f in FRACTIONS for s in SEEDS]

def write_table(df, out, stem):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'{stem}.csv', index=False)
    try: (out / f'{stem}.md').write_text(df.to_markdown(index=False, floatfmt='.5f'))
    except ImportError: (out / f'{stem}.md').write_text(df.to_string(index=False))
    (out / f'{stem}.tex').write_text(df.to_latex(index=False, float_format=lambda x: f'{x:.5f}', escape=True))


def code_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()
