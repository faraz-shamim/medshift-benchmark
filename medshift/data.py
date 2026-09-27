from __future__ import annotations
import concurrent.futures, hashlib, io, json, math, os, re, shutil, tarfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm
from .common import *

# Kaggle mirrors are delivery mechanisms, not grants of rights. Keep derived outputs private.
# Version 1 is intentionally requested: do not silently switch data versions during a study.
HANDLES = {
 'nih': 'jbeltranleon/nih-chest-xrays-224-gray/versions/1',
 'chexpert': 'minatozaki/chexpert-14/versions/1',
 'vinbig': 'chhengli/vinbigdata-512x512-t1/versions/1',
}
ALTERNATIVES = {
 'nih': 'khanfashee/nih-chest-x-ray-14-224x224-resized',
 'vinbig': 'akilason/vinbigdata-640-image-dataset/versions/1',
}
# Small reference metadata only, used if the image mirror omits its CSV.
# The resolved content SHA256 is frozen into provenance; further runs must use the saved output.
REFERENCE_CSV = {
 'nih': 'https://raw.githubusercontent.com/mlmed/torchxrayvision/main/torchxrayvision/data/Data_Entry_2017_v2020.csv.gz',
 'vinbig': 'https://raw.githubusercontent.com/mlmed/torchxrayvision/main/torchxrayvision/data/vinbigdata-train.csv.gz',
}

class ImageIndex:
    """Index image names in directories and archives without extracting full archives.

    ZIP and uncompressed TAR support random reads. Compressed TAR is streamed once
    to small uint8 arrays by materialize_compressed_archives before indexing.
    Ambiguous basenames are NOT guessed; CheXpert is resolved by full patient path.
    """
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.jp2'}
    def __init__(self, root):
        self.root = Path(root); self.full = {}; self.base = {}
        for p in self.root.rglob('*'):
            if not p.is_file(): continue
            if p.suffix.lower() in self.suffixes:
                self.add(str(p.relative_to(self.root)).replace('\\','/'), ('file', str(p), ''))
            elif p.suffix.lower() == '.zip':
                with zipfile.ZipFile(p) as z:
                    for name in z.namelist():
                        if Path(name).suffix.lower() in self.suffixes:
                            self.add(name, ('zip', str(p), name))
            elif p.suffix.lower() == '.tar':
                with tarfile.open(p) as t:
                    for member in t:
                        if member.isfile() and Path(member.name).suffix.lower() in self.suffixes:
                            self.add(member.name, ('tar', str(p), member.name))
        if not self.full:
            raise RuntimeError('No PNG/JPEG/JP2 images or supported archives found. Inspect the printed file list. '
                               'Do not guess the row order of an anonymous HDF5/NPY mirror; use a named-image mirror.')
    def add(self, name, loc):
        name = name.lstrip('./').replace('\\','/')
        if name in self.full and self.full[name] != loc: raise ValueError(f'Duplicate relative image path: {name}')
        self.full[name] = loc
        self.base.setdefault(Path(name).stem, []).append(loc)
        # Canonical CheXpert suffix is independent of mirror wrapper directories.
        match = re.search(r'(?:^|/)((?:train|valid)/patient\d+/.*)$', name)
        if match:
            key = match.group(1)
            if key in self.full and self.full[key] != loc: raise ValueError(f'Ambiguous CheXpert path {key}')
            self.full[key] = loc
    def resolve(self, name):
        name = str(name).replace('\\','/').lstrip('./')
        match = re.search(r'(?:^|/)((?:train|valid)/patient\d+/.*)$', name)
        if match and match.group(1) in self.full: return self.full[match.group(1)]
        if name in self.full: return self.full[name]
        locs = list(dict.fromkeys(self.base.get(Path(name).stem, [])))
        if len(locs) == 1: return locs[0]
        if not locs: raise FileNotFoundError(name)
        raise ValueError(f'Ambiguous basename {name}: {len(locs)} possible files.')


def materialize_compressed_archives(root, scratch):
    """One-image-at-a-time conversion; never keeps a second full-resolution archive copy."""
    root, scratch = Path(root), Path(scratch)
    archives = list(root.rglob('*.tar.gz')) + list(root.rglob('*.tgz'))
    if not archives: return root
    # Most selected mirrors are loose images. This is a deliberate compatibility route.
    scratch.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        done = scratch / (stable_hash(str(archive))[:16] + '.done')
        if done.exists(): continue
        with tarfile.open(archive, mode='r|*') as tf:
            for member in tf:
                if not member.isfile() or Path(member.name).suffix.lower() not in ImageIndex.suffixes: continue
                rel = Path(member.name)
                if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe archive member path.')
                dest = scratch / rel; dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists(): continue
                data = tf.extractfile(member).read()
                with Image.open(io.BytesIO(data)) as im:
                    im.convert('L').resize((224,224), Image.Resampling.BILINEAR).save(dest.with_suffix('.png'))
        done.write_text('complete')
    for csv in list(root.rglob('*.csv')) + list(root.rglob('*.csv.gz')):
        shutil.copy2(csv, scratch / csv.name)
    return scratch


def get_data_root(domain, handle=None, explicit_root=None, licenses_reviewed=False):
    if not licenses_reviewed:
        raise PermissionError('Review the original data terms, obtain any required permission, keep outputs PRIVATE, '
                              'then set LICENSES_REVIEWED=True. A mirror is not a replacement for a license.')
    if explicit_root:
        root = Path(explicit_root)
        if not root.exists(): raise FileNotFoundError(root)
        return root, {'handle': handle or HANDLES[domain], 'delivery': 'explicit_attached_input'}
    import kagglehub
    handle = handle or HANDLES[domain]
    # Inside Kaggle, this attaches a shared, read-only dataset; do not pass output_dir=working.
    root = Path(kagglehub.dataset_download(handle))
    return root, {'handle': handle, 'delivery': 'kagglehub', 'resolved_root': str(root)}


def find_csv(root, columns):
    matches = []
    for p in list(Path(root).rglob('*.csv')) + list(Path(root).rglob('*.csv.gz')):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
            if set(columns).issubset(cols): matches.append(p)
        except (ValueError, OSError, pd.errors.ParserError): continue
    return matches


def metadata_csv(domain, root, out):
    required = {'nih':['Image Index','Finding Labels','Patient ID'],
                'chexpert':['Path','Frontal/Lateral','Cardiomegaly','Pleural Effusion'],
                'vinbig':['image_id','class_name','rad_id']}[domain]
    candidates = find_csv(root, required)
    if domain == 'chexpert':
        candidates = [p for p in candidates if p.name.lower() == 'train.csv']
    if candidates:
        # Different copies are allowed only when their file hashes agree.
        groups = {}
        for p in candidates: groups.setdefault(sha256_file(p), []).append(p)
        if len(groups) > 1:
            raise RuntimeError(f'Multiple non-identical metadata files: {candidates}. Supply a clean input.')
        src = candidates[0]
        dest = Path(out) / ('source_metadata' + ''.join(src.suffixes))
        shutil.copy2(src, dest)
        return dest, {'metadata_origin': str(src), 'metadata_sha256': sha256_file(dest)}
    if domain not in REFERENCE_CSV:
        raise FileNotFoundError('The CheXpert image dataset must include its unmodified train.csv.')
    url = REFERENCE_CSV[domain]
    dest = Path(out) / 'source_metadata.csv.gz'
    r = requests.get(url, timeout=(15,120)); r.raise_for_status()
    if len(r.content) > 30_000_000: raise RuntimeError('Unexpectedly large reference metadata response.')
    dest.write_bytes(r.content)
    return dest, {'metadata_origin':url, 'metadata_sha256':sha256_file(dest),
                  'metadata_note':'Reference metadata from TorchXRayVision; content frozen by SHA256.'}


def numeric_age(series):
    # Invalid/ambiguous ages are excluded rather than guessed from strings such as 020M.
    return pd.to_numeric(series, errors='coerce')


def build_records(domain, metadata, test_mode=False):
    d = pd.read_csv(metadata)
    flow = [{'stage':'metadata_rows', 'n':len(d)}]
    if domain == 'nih':
        if not test_mode and d['Image Index'].nunique() != 112120:
            raise ValueError('Expected metadata for all 112,120 NIH images; a 5% sample is not the protocol dataset.')
        age = numeric_age(d['Patient Age'])
        d = d[age.between(18,100) & d['View Position'].isin(['AP','PA'])].copy()
        d['age'] = numeric_age(d['Patient Age'])
        r = pd.DataFrame({'original_id':d['Image Index'].astype(str),
                          'patient_id':'nih_' + d['Patient ID'].astype(str),
                          'image_uid':'nih_' + d['Image Index'].astype(str),
                          'sex':d['Patient Gender'].replace({'M':'Male','F':'Female'}),
                          'age':d['age'], 'view':d['View Position']})
        tokens = d['Finding Labels'].fillna('').str.split('|')
        for label, token in zip(LABELS, ['Cardiomegaly','Effusion']):
            r[label] = tokens.apply(lambda x: int(token in x))
            r[label+'_raw'] = r[label]
        r['patient_id_known'] = True
    elif domain == 'chexpert':
        if not test_mode and d.Path.nunique() < 200000:
            raise ValueError('Expected the full CheXpert-small training manifest, not a balanced/sample mirror.')
        age = numeric_age(d.Age)
        d = d[(d['Frontal/Lateral'] == 'Frontal') & age.between(18,100)].copy()
        d['patient_id'] = d.Path.str.extract(r'(patient\d+)')[0].values
        if d.patient_id.isna().any(): raise ValueError('Cannot recover CheXpert patient identifiers.')
        # One image per selected patient, selected without examining outcome labels.
        d['_order'] = d.Path.map(lambda p: stable_hash(p, PROTOCOL['split_seed']))
        d = d.sort_values('_order').drop_duplicates('patient_id')
        d = d.assign(_patient_order=d.patient_id.map(lambda p: stable_hash(p, PROTOCOL['split_seed'])))
        d = d.sort_values('_patient_order').head(PROTOCOL['chexpert_patient_cap']).copy()
        if not test_mode and len(d) != PROTOCOL['chexpert_patient_cap']:
            raise ValueError('Fewer than 20,000 eligible CheXpert patients; do not silently change the cohort.')
        r = pd.DataFrame({'original_id':d.Path, 'patient_id':'chexpert_' + d.patient_id,
                          'image_uid':d.Path.map(lambda p:'chexpert_'+stable_hash(p)[:24]),
                          'sex':d.Sex, 'age':numeric_age(d.Age), 'view':d['AP/PA']})
        for label, col in zip(LABELS, ['Cardiomegaly','Pleural Effusion']):
            raw = pd.to_numeric(d[col], errors='coerce')
            if not set(raw.dropna().unique()).issubset({-1,0,1}): raise ValueError(f'Invalid {col} labels.')
            r[label+'_raw'] = raw
            r[label] = raw.fillna(0).replace(-1,np.nan)
        r['patient_id_known'] = True
    elif domain == 'vinbig':
        if not test_mode and d.image_id.nunique() != 15000:
            raise ValueError('Require 15,000 labeled VinBigData training images, NOT the unlabeled competition test set.')
        readers = d.groupby('image_id').rad_id.nunique()
        if not (readers == 3).all():
            raise ValueError('Every VinBig image must retain all 3 radiologist IDs; WBF/aggregated CSVs are unsuitable.')
        ids = sorted(d.image_id.unique())
        r = pd.DataFrame({'original_id':ids, 'image_uid':['vinbig_'+i for i in ids],
                          'patient_id':['vinbig_image_'+i for i in ids],
                          'patient_id_known':False, 'sex':'Unknown', 'age':np.nan, 'view':'PA'})
        for label, name in zip(LABELS, ['Cardiomegaly','Pleural effusion']):
            votes = d[d.class_name.str.casefold() == name.casefold()].drop_duplicates(['image_id','rad_id']).groupby('image_id').size()
            count = r.original_id.map(votes).fillna(0).astype(int)
            r[label+'_votes'] = count
            r[label] = (count >= 2).astype(int)
            r[label+'_raw'] = r[label]
    else: raise ValueError(domain)
    r = r.reset_index(drop=True)
    r['domain'] = domain
    if not r.image_uid.is_unique: raise ValueError('Duplicate IDs in source metadata.')
    flow.append({'stage':'eligible_selected_before_pixel_audit', 'n':len(r)})
    return r.sort_values('image_uid').reset_index(drop=True), flow


def read_location(loc):
    kind, file, member = loc
    if kind == 'file':
        with open(file, 'rb') as f: return f.read()
    if kind == 'zip':
        with zipfile.ZipFile(file) as z: return z.read(member)
    if kind == 'tar':
        with tarfile.open(file) as t: return t.extractfile(member).read()
    raise ValueError(kind)


def decode_record(item):
    row, loc = item
    try:
        raw = read_location(loc)
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
            if min(width,height) < 100: raise ValueError('Image dimension <100 pixels.')
            if im.mode.startswith('I') or im.mode == 'F':
                raise ValueError('Unexpected high-bit-depth image; require documented 8-bit PNG/JPEG rendering.')
            gray = im.convert('L')
            small = gray.resize((224,224), Image.Resampling.BILINEAR)
            pixels = np.asarray(small, dtype=np.uint8)
            if float(pixels.std()) < 1.0: raise ValueError('Near-constant image.')
            h = np.asarray(gray.resize((9,8), Image.Resampling.BILINEAR))
            bits = (h[:,1:] > h[:,:-1]).reshape(-1)
            dhash = f'{int.from_bytes(np.packbits(bits).tobytes(),"big"):016x}'
        row.update(original_width=width, original_height=height,
                   pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                   source_file_sha256=hashlib.sha256(raw).hexdigest(), dhash=dhash)
        return row, pixels, None
    except Exception as e:
        return row, None, f'{type(e).__name__}: {e}'


def split_nih(r):
    ids = sorted(r.patient_id.unique(), key=lambda x:stable_hash(x,PROTOCOL['split_seed']))
    n=len(ids); a=int(n*.6); b=a+int(n*.1); c=b+int(n*.1)
    labels=['train']*a+['tune']*(b-a)+['calibration']*(c-b)+['test']*(n-c)
    mapping=dict(zip(ids,labels))
    r=r.copy(); r['split']=r.patient_id.map(mapping)
    assert_patient_disjoint(r)
    return r


def prepare_dataset(domain, output_parent='/kaggle/working', *, licenses_reviewed=False,
                    handle=None, explicit_root=None, resume_from=None, workers=4,
                    max_hours=7.0, test_mode=False):
    """Run once per dataset (CPU). Output is <= about 5.7 GB of uint8 NIH data."""
    out=Path(output_parent)/f'prepared_{domain}'; out.mkdir(parents=True, exist_ok=True)
    if resume_from and not list(out.glob('chunks/*.csv')):
        shutil.copytree(resume_from,out,dirs_exist_ok=True)
    if (out/'COMPLETE.json').exists():
        _, info=load_prepared(out); print('Already complete:',info); return out
    started=time.monotonic()
    root, provenance=get_data_root(domain,handle,explicit_root,licenses_reviewed)
    print('Input:',root)
    print('Example input files:',[str(p.relative_to(root)) for p in list(root.iterdir())[:15]])
    input_bytes=directory_bytes(root)
    # Attached inputs are not saved-output storage; this guard also honors the requested per-input budget.
    if input_bytes > 19e9:
        raise RuntimeError(f'Input is {input_bytes/1e9:.1f} GB. Use the small/224 mirror, not original DICOM/full-resolution data.')
    provenance.update(input_bytes=input_bytes, licenses_reviewed=True, keep_outputs_private=True)
    scratch=Path('/tmp')/f'medshift_expanded_{domain}'
    indexed_root=materialize_compressed_archives(root,scratch)
    meta, meta_info=metadata_csv(domain,indexed_root,out); provenance.update(meta_info)
    records,flow=build_records(domain,meta,test_mode)
    request_hash=hashlib.sha256((provenance['metadata_sha256']+str(provenance['handle'])+str(len(records))+VERSION).encode()).hexdigest()
    if (out/'request.json').exists() and read_json(out/'request.json')['sha256'] != request_hash:
        raise RuntimeError('Inputs changed since partial preparation. Start a separate version; do not mix shards.')
    write_json(out/'request.json',{'sha256':request_hash})
    index=ImageIndex(indexed_root)
    (out/'chunks').mkdir(exist_ok=True); (out/'shards').mkdir(exist_ok=True)
    chunk_size=2048
    for start in range(0,len(records),chunk_size):
        k=start//chunk_size; chunk_csv=out/'chunks'/f'{k:04d}.csv'; shard=out/'shards'/f'{k:04d}.npy'
        if chunk_csv.exists() and shard.exists(): continue
        if time.monotonic()-started > max_hours*3600:
            write_json(out/'INCOMPLETE.json',{'next_record':start,'total_records':len(records)})
            print('Preparation paused safely. Save PRIVATE output, attach it, set RESUME_FROM, and rerun.'); return out
        pairs=[]
        for row in records.iloc[start:start+chunk_size].to_dict('records'):
            # Missing images are a wrong/incomplete mirror, NOT a clinical exclusion.
            try: loc=index.resolve(row['original_id'])
            except Exception as e: raise RuntimeError(f'Metadata/image mismatch at {row["original_id"]}: {e}') from e
            pairs.append((row,loc))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            decoded=list(tqdm(pool.map(decode_record,pairs),total=len(pairs),desc=f'{domain} shard {k}'))
        arr=np.zeros((len(decoded),224,224),dtype=np.uint8); rows=[]
        for j,(row,pixels,error) in enumerate(decoded):
            row.update(shard=f'shards/{k:04d}.npy', offset=j, decode_error=error or '')
            if pixels is not None: arr[j]=pixels
            rows.append(row)
        tmp=shard.with_suffix('.npy.tmp')
        with open(tmp,'wb') as f: np.save(f,arr,allow_pickle=False)
        tmp.replace(shard)
        pd.DataFrame(rows).to_csv(chunk_csv,index=False)
        check_storage(output_parent,reserve_gb=.5)
    r=pd.concat([pd.read_csv(p,dtype={'patient_id':str,'image_uid':str,'dhash':str}) for p in sorted((out/'chunks').glob('*.csv'))],ignore_index=True)
    corrupt=r[r.decode_error.notna() & r.decode_error.ne('')]
    corrupt.to_csv(out/'excluded_corrupt.csv',index=False)
    if len(corrupt)/max(len(r),1) > .001:
        raise RuntimeError(f'{len(corrupt)} corrupt/unexpected images (>0.1%); inspect rendering rather than silently excluding them.')
    r=r[r.decode_error.isna() | r.decode_error.eq('')].copy()
    duplicates=r[r.duplicated('pixel_sha256',keep=False)].copy()
    duplicates.to_csv(out/'exact_pixel_duplicates.csv',index=False)
    # All images in a cross-patient exact-duplicate group are removed before partitioning.
    cross_patient=r.groupby('pixel_sha256').patient_id.nunique()
    ambiguous=set(cross_patient[cross_patient>1].index)
    r=r[~r.pixel_sha256.isin(ambiguous)].drop_duplicates('pixel_sha256',keep='first').copy()
    flow.extend([{'stage':'corrupt_exclusions','n':len(corrupt)},
                 {'stage':'after_exact_pixel_dedup','n':len(r)}])
    r=split_nih(r) if domain=='nih' else r.assign(split='test')
    r=r.sort_values('image_uid').reset_index(drop=True)
    r.to_csv(out/'manifest.csv',index=False)
    pd.DataFrame(flow).to_csv(out/'cohort_flow.csv',index=False)
    write_json(out/'provenance.json',provenance)
    environment(out/'environment.json')
    npy_hashes={str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'shards').glob('*.npy'))}
    write_json(out/'shard_checksums.json',npy_hashes)
    info={'domain':domain,'images':len(r),'unique_groups':r.patient_id.nunique(),
          'patient_ids_available':domain!='vinbig','manifest_sha256':sha256_file(out/'manifest.csv'),
          'prepared_bytes':directory_bytes(out),'protocol_sha256':protocol_hash(),'test_mode':test_mode}
    write_json(out/'COMPLETE.json',info)
    if (out/'INCOMPLETE.json').exists(): (out/'INCOMPLETE.json').unlink()
    print('COMPLETE',info)
    return out


def audit_all(prepared_roots, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    data={}
    for domain,root in prepared_roots.items():
        frame,info=load_prepared(root)
        if info.get('test_mode'): raise RuntimeError('Synthetic fixtures cannot be locked as research data.')
        if info['protocol_sha256']!=protocol_hash(): raise RuntimeError('Prepared data use a different protocol; rerun preparation.')
        for relative,digest in read_json(Path(root)/'shard_checksums.json').items():
            if sha256_file(Path(root)/relative)!=digest: raise RuntimeError(f'Shard integrity failure: {relative}')
        data[domain]=frame
    all_rows=pd.concat(data.values(),ignore_index=True)
    cross=all_rows.groupby('pixel_sha256').domain.nunique()
    exact=set(cross[cross>1].index)
    matches=all_rows[all_rows.pixel_sha256.isin(exact)]
    matches.to_csv(out/'cross_dataset_exact_matches.csv',index=False)
    # Exact cross-domain matches invalidate independent evaluation: fail, do not cherry-pick exclusions later.
    if exact: raise RuntimeError('Exact cross-dataset image matches detected. Resolve provenance and re-register exclusions before training.')
    # dHash equality is only a screening tool. It neither proves nor excludes near-duplication.
    near=all_rows[all_rows.duplicated('dhash',keep=False)].copy()
    near.to_csv(out/'dhash_review_candidates.csv',index=False)
    assert_patient_disjoint(data['nih'])
    summaries=[]
    for domain,df in data.items():
        for split,g in df.groupby('split'):
            for label in LABELS:
                known=g[label].notna(); yy=g.loc[known,label]
                summaries.append({'domain':domain,'split':split,'label':label,'images':len(g),
                                  'patients_or_image_groups':g.patient_id.nunique(),'known':int(known.sum()),
                                  'positive':int(yy.sum()),'negative':int((yy==0).sum()),'masked':int((~known).sum()),
                                  'prevalence_known':float(yy.mean()),'median_age':g.age.median(),
                                  'female_n':int((g.sex=='Female').sum()),'PA_n':int((g.view=='PA').sum())})
    write_table(pd.DataFrame(summaries),out,'T1_cohort')
    write_json(out/'audit.json',{'exact_cross_dataset_matches':len(matches),'dhash_candidates':len(near),
                                'warning':'dHash candidates require human review; absence of exact matches does not prove patient independence.',
                                'vinbig_resampling_unit':'image; public mirror does not expose patient IDs'})
    return data
