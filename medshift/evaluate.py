from __future__ import annotations
import shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import *
from .train import (Classifier, make_loader, autocast, seed_everything, find_study_lock,
                    pause_requested, training_signature)


def collect_runs(search_root='/kaggle/input'):
    candidates=list(Path(search_root).rglob('medshift_runs/*/done.json'))
    result={}
    for p in candidates:
        done=read_json(p)
        if done.get('pilot'): continue
        jid=p.parent.name
        if jid in result:
            old=read_json(result[jid]/'done.json')
            if old['best_checkpoint_sha256']!=done['best_checkpoint_sha256']:
                raise RuntimeError(f'Different completed checkpoints attached for {jid}.')
        result[jid]=p.parent
    missing=[j['job_id'] for j in jobs() if j['job_id'] not in result]
    if missing: raise RuntimeError(f'Complete all 36 training jobs before external evaluation. Missing: {missing}')
    return result


def restore_predictions(out,search_root='/kaggle/input'):
    out=Path(out)
    for src in Path(search_root).rglob('predictions/*/*.npz'):
        dest=out/'predictions'/src.parent.name/src.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            with np.load(src,allow_pickle=False) as a,np.load(dest,allow_pickle=False) as b:
                if str(a['signature'])!=str(b['signature']): raise RuntimeError('Conflicting inference cache signatures.')
        else: shutil.copy2(src,dest)


@torch.no_grad()
def predict(model,df,path,signature,*,device,mc_samples=0,workers=2,seed=0,
            deadline=None,stop_file=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as existing:
            if str(existing['signature'])!=signature or not np.array_equal(existing['image_uid'],df.image_uid.astype(str).to_numpy()):
                raise RuntimeError('Cached predictions do not match model or dataset.')
        return True
    seed_everything(seed); model.eval(); chunks=[]; mc_chunks=[]
    loader=make_loader(df,batch=64,workers=workers)
    start=time.monotonic()
    for x,_,_ in loader:
        if pause_requested(deadline,stop_file):
            print('SAFE PAUSE: unfinished prediction context will be recomputed:',path.name,flush=True)
            return False
        x=x.to(device,non_blocking=True)
        with autocast(device):
            features=model.backbone(x)
            logits=model.head(features)  # dropout is disabled for deterministic inference
            if mc_samples:
                # Trained head-only dropout, NOT whole-network Bayesian inference.
                model.dropout.train(True)
                draws=torch.stack([model.head(model.dropout(features)) for _ in range(mc_samples)],dim=1)
                model.dropout.train(False)
        chunks.append(logits.float().cpu().numpy())
        if mc_samples: mc_chunks.append(draws.float().cpu().numpy())
    arrays={'image_uid':df.image_uid.astype(str).to_numpy(dtype=str),'logits':np.concatenate(chunks),
            'signature':np.asarray(signature),'inference_seconds':np.asarray(time.monotonic()-start)}
    if mc_samples: arrays['mc_logits']=np.concatenate(mc_chunks)
    tmp=path.with_suffix('.npz.tmp')
    with open(tmp,'wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)
    return True


def prediction_signature(job,done,context,domain,lock):
    mc=PROTOCOL['mc_head_samples'] if job['fraction']==1 else 0
    return stable_hash(done['best_checkpoint_sha256']+lock['datasets'][domain]+context+str(mc)+VERSION+code_hash())


def make_contexts(prepared_roots,lock):
    data={}; infos={}
    if set(prepared_roots)!=set(DOMAINS):
        raise RuntimeError('Evaluation requires NIH, CheXpert and VinBig prepared outputs.')
    for domain,root in prepared_roots.items():
        data[domain],infos[domain]=load_prepared(root)
        if infos[domain]['manifest_sha256']!=lock['datasets'][domain]:
            raise RuntimeError(f'{domain} differs from the locked dataset.')
        if infos[domain].get('test_mode'):
            raise RuntimeError('Synthetic data cannot enter manuscript inference.')
    contexts={'nih_calibration':data['nih'][data['nih'].split=='calibration'].copy(),
              'nih_test':data['nih'][data['nih'].split=='test'].copy(),
              'chexpert_test':data['chexpert'],'vinbig_test':data['vinbig']}
    for context,df in contexts.items():
        if df.empty: raise RuntimeError(f'Empty evaluation cohort: {context}')
        contexts[context]=df.sort_values('image_uid').reset_index(drop=True)
    return contexts,infos


def validate_training_record(job,run,source_manifest_sha256):
    run=Path(run); cfg=read_json(run/'run.json'); done=read_json(run/'done.json')
    sig=training_signature(job,source_manifest_sha256,pilot=False)
    if cfg.get('signature')!=sig or done.get('signature')!=sig or done.get('pilot'):
        raise RuntimeError(f'Frozen training code/data signature mismatch: {job["job_id"]}.')
    if cfg.get('source_manifest_sha256')!=source_manifest_sha256:
        raise RuntimeError('Training used a different source manifest.')
    if sha256_file(run/'best.pt')!=done['best_checkpoint_sha256']:
        raise RuntimeError('Checkpoint checksum mismatch.')
    return cfg,done


def prepare_evaluation(prepared_roots,output,search_root='/kaggle/input'):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    lock=find_study_lock(search_root); runs=collect_runs(search_root)
    contexts,infos=make_contexts(prepared_roots,lock)
    (out/'metadata').mkdir(exist_ok=True)
    for context,df in contexts.items():
        df.drop(columns=['_root'],errors='ignore').to_csv(out/'metadata'/f'{context}.csv',index=False)
    # Check every checkpoint BEFORE launching either external-evaluation worker.
    for job in jobs():
        run=runs[job['job_id']]
        validate_training_record(job,run,infos['nih']['manifest_sha256'])
        meta=out/'training_records'/job['job_id']; meta.mkdir(parents=True,exist_ok=True)
        for name in ['run.json','done.json','history.csv']:
            shutil.copy2(run/name,meta/name)
    restore_predictions(out,search_root)
    write_json(out/'study_lock.json',lock); write_json(out/'protocol.json',PROTOCOL)
    return out,lock,runs,contexts,infos


def valid_prediction(path,job,done,context,df,lock):
    path=Path(path)
    if not path.exists(): return False
    sig=prediction_signature(job,done,context,df.domain.iloc[0],lock)
    with np.load(path,allow_pickle=False) as saved:
        if str(saved['signature'])!=sig or not np.array_equal(saved['image_uid'],df.image_uid.astype(str).to_numpy()):
            raise RuntimeError(f'Stale/corrupt prediction cache: {path}. Do not mix old-code outputs.')
        if saved['logits'].shape!=(len(df),len(LABELS)) or not np.isfinite(saved['logits']).all():
            raise RuntimeError(f'Invalid logits: {path}')
        if job['fraction']==1:
            expected=(len(df),PROTOCOL['mc_head_samples'],len(LABELS))
            if 'mc_logits' not in saved or saved['mc_logits'].shape!=expected or not np.isfinite(saved['mc_logits']).all():
                raise RuntimeError(f'Invalid/missing MC head predictions: {path}')
    return True


def job_predictions_complete(job,run,contexts,lock,out):
    done=read_json(Path(run)/'done.json')
    # Do not short-circuit: validate every existing file even if an earlier one is absent.
    checks=[valid_prediction(Path(out)/'predictions'/job['job_id']/f'{context}.npz',
                             job,done,context,df,lock) for context,df in contexts.items()]
    return all(checks)


def inference_one(job,run,prepared_roots,out,*,deadline,workers=1,stop_file=None):
    """One frozen model, all four contexts, on this process's isolated cuda:0."""
    out=Path(out); run=Path(run); lock=read_json(out/'study_lock.json')
    if lock.get('code_sha256')!=code_hash(): raise RuntimeError('Worker code differs from lock.')
    contexts,infos=make_contexts(prepared_roots,lock)
    cfg,done=validate_training_record(job,run,infos['nih']['manifest_sha256'])
    device=torch.device('cuda:0')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('An inference worker must see exactly one isolated CUDA device.')
    model=None; mc=PROTOCOL['mc_head_samples'] if job['fraction']==1 else 0
    try:
        for ci,(context,df) in enumerate(contexts.items()):
            path=out/'predictions'/job['job_id']/f'{context}.npz'
            if valid_prediction(path,job,done,context,df,lock): continue
            if pause_requested(deadline,stop_file) or time.monotonic()+120>deadline:
                return False
            sig=prediction_signature(job,done,context,df.domain.iloc[0],lock)
            if model is None:
                model=Classifier(job['model'],pretrained=False)
                ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(ck['model']); model=model.to(device)
            finished=predict(model,df,path,sig,device=device,mc_samples=mc,workers=workers,
                             seed=job['seed']*100+ci,deadline=deadline,stop_file=stop_file)
            if not finished: return False
            print('Predicted',job['job_id'],context,len(df),'MC head samples',mc,flush=True)
    finally:
        del model; torch.cuda.empty_cache()
    return job_predictions_complete(job,run,contexts,lock,out)


def evaluation_status(out,runs,contexts,lock):
    out=Path(out); rows=[]; hashes={}; metadata_hashes={}
    for path in sorted((out/'metadata').glob('*.csv')):
        metadata_hashes[str(path.relative_to(out))]=sha256_file(path)
    for job in jobs():
        done=read_json(runs[job['job_id']]/'done.json')
        for context,df in contexts.items():
            path=out/'predictions'/job['job_id']/f'{context}.npz'
            complete=valid_prediction(path,job,done,context,df,lock)
            if complete: hashes[str(path.relative_to(out))]=sha256_file(path)
            rows.append({'job_id':job['job_id'],'context':context,'complete':complete})
    status=pd.DataFrame(rows); status.to_csv(out/'inference_status.csv',index=False)
    return status,hashes,metadata_hashes


def inference_queue(*args,**kwargs):
    """Backward API alias; new notebook calls inference_queue_parallel explicitly."""
    from .parallel import inference_queue_parallel
    if 'workers' in kwargs: kwargs['workers_per_gpu']=kwargs.pop('workers')
    return inference_queue_parallel(*args,**kwargs)
