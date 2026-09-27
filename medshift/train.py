from __future__ import annotations
import contextlib, copy, hashlib, math, os, random, shutil, time
import fcntl
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .common import *


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def worker_seed(worker_id):
    seed=torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


class ShardDataset(Dataset):
    def __init__(self, df, augment=False):
        from torchvision import transforms as T
        self.df=df.reset_index(drop=True); self.cache=OrderedDict()
        ops=[]
        if augment:
            ops=[T.RandomAffine(degrees=5,translate=(.02,.02),interpolation=T.InterpolationMode.BILINEAR,fill=0),
                 T.ColorJitter(brightness=.1,contrast=.1)]
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]; path=str(Path(row['_root'])/row.shard)
        if path not in self.cache:
            self.cache[path]=np.load(path,mmap_mode='r',allow_pickle=False)
            if len(self.cache)>4: self.cache.popitem(last=False)
        arr=np.array(self.cache[path][int(row.offset)],copy=True)
        x=self.transform(Image.fromarray(arr).convert('RGB'))
        y=torch.tensor(row[LABELS].to_numpy(dtype=np.float32))
        return x,y,i


class Classifier(nn.Module):
    def __init__(self,name,pretrained=True,backbone_override=None):
        super().__init__()
        if backbone_override is None:
            import timm
            self.backbone=timm.create_model(MODELS[name],pretrained=pretrained,num_classes=0)
        else: self.backbone=backbone_override
        self.dropout=nn.Dropout(PROTOCOL['head_dropout'])
        self.head=nn.Linear(self.backbone.num_features,len(LABELS))
    def forward(self,x): return self.head(self.dropout(self.backbone(x)))


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(name,root,pretrained=True):
    """Serialize shared initial-weight cache access across independent GPU jobs."""
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    cache=root/f'{name}.pt'
    # OS advisory locks are released automatically when a worker exits/crashes.
    # A single shared cache means all fractions/seeds use the same starting backbone.
    with open(root/f'{name}.cache.lock','a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        model=Classifier(name,pretrained=pretrained and not cache.exists())
        if cache.exists():
            content=torch.load(cache,map_location='cpu',weights_only=True)
            model.backbone.load_state_dict(content['state'])
            digest=state_hash(content['state'])
            if digest!=content['sha256']:
                raise RuntimeError('Initial pretrained-weight cache checksum mismatch.')
        else:
            state={k:v.detach().cpu() for k,v in model.backbone.state_dict().items()}
            digest=state_hash(state)
            atomic_torch_save({'state':state,'sha256':digest},cache)
    return model,digest


def atomic_torch_save(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(data,tmp); tmp.replace(path)


def make_loader(df,*,train=False,batch=32,workers=2,seed=0):
    gen=torch.Generator().manual_seed(seed)
    return DataLoader(ShardDataset(df,augment=train),batch_size=batch,shuffle=train,
                      num_workers=workers,pin_memory=torch.cuda.is_available(),drop_last=False,
                      worker_init_fn=worker_seed,generator=gen,persistent_workers=False,
                      multiprocessing_context='spawn' if workers else None)


def masked_bce(logits,y):
    mask=torch.isfinite(y)
    values=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.nan_to_num(y),reduction='none')
    return values[mask].mean()


def autocast(device):
    return torch.autocast(device_type='cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()


class PauseRequested(RuntimeError):
    """Internal safe-stop signal; incomplete epochs/contexts are recomputed."""


def pause_requested(deadline=None,stop_file=None):
    return ((deadline is not None and time.monotonic() >= deadline) or
            (stop_file is not None and Path(stop_file).exists()))


@torch.no_grad()
def validation_loss(model,loader,device,*,deadline=None,stop_file=None):
    model.eval(); total=0.; count=0
    for x,y,_ in loader:
        if pause_requested(deadline,stop_file):
            raise PauseRequested('Pause during validation; keep the previous completed epoch.')
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with autocast(device): logits=model(x)
        mask=torch.isfinite(y)
        vals=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),torch.nan_to_num(y),reduction='none')
        total+=float(vals[mask].sum()); count+=int(mask.sum())
    if count==0: raise RuntimeError('No observed tuning labels.')
    return total/count


def training_signature(job, manifest_sha256, pilot=False):
    return stable_hash(json.dumps({'protocol':protocol_hash(),'data':manifest_sha256,
                                  'job':job,'pilot':pilot,'code_sha256':code_hash()},sort_keys=True))


def validate_run(job, source_info, out, pilot=False):
    """Reject old-code/pilot/incompatible outputs even when they are already done."""
    run=Path(out)/'medshift_runs'/job['job_id']
    signature=training_signature(job,source_info['manifest_sha256'],pilot)
    cfg_path=run/'run.json'
    if not cfg_path.exists():
        if (run/'done.json').exists() or (run/'last.pt').exists():
            raise RuntimeError(f'Run metadata missing for {job["job_id"]}; do not reuse this output.')
        return False
    cfg=read_json(cfg_path)
    if cfg.get('signature')!=signature or bool(cfg.get('pilot'))!=bool(pilot):
        raise RuntimeError(f'Incompatible code/data/protocol or pilot checkpoint: {job["job_id"]}. '
                           'Attach only outputs from the current dual-GPU release and the new lock.')
    if (run/'done.json').exists():
        done=read_json(run/'done.json')
        if done.get('signature')!=signature:
            raise RuntimeError(f'Completion signature mismatch: {job["job_id"]}.')
        if not (run/'best.pt').exists() or sha256_file(run/'best.pt')!=done.get('best_checkpoint_sha256'):
            raise RuntimeError(f'Completed checkpoint checksum mismatch: {job["job_id"]}.')
        return True
    return False


def copy_prior_runs(out,search_root='/kaggle/input'):
    """Cumulative checkpoints. Copy ONLY private outputs made by this package."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    candidates=list(Path(search_root).rglob('medshift_runs/*/run.json'))
    by_id={}
    for p in candidates: by_id.setdefault(p.parent.name,[]).append(p.parent)
    for name,paths in by_id.items():
        target=out/'medshift_runs'/name
        existing=paths+([target] if (target/'run.json').exists() else [])
        signatures={read_json(p/'run.json')['signature'] for p in existing}
        if len(signatures)>1: raise RuntimeError(f'Conflicting attached runs for {name}; remove old-protocol/pilot inputs.')
        def progress(p):
            if (p/'done.json').exists(): return (2,read_json(p/'done.json')['epochs_run'])
            return (1,len(pd.read_csv(p/'history.csv')) if (p/'history.csv').exists() else 0)
        chosen=max(existing,key=progress)
        if chosen!=target: shutil.copytree(chosen,target,dirs_exist_ok=True)
    for p in Path(search_root).rglob('initial_weights/*.pt'):
        dest=out/'initial_weights'/p.name; dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            if sha256_file(p)!=sha256_file(dest):
                # Serialization bytes can differ; the tensor-state hash must agree.
                a=torch.load(p,map_location='cpu',weights_only=True); b=torch.load(dest,map_location='cpu',weights_only=True)
                if a['sha256']!=b['sha256']: raise RuntimeError('Conflicting initial pretrained model weights.')
        else: shutil.copy2(p,dest)


def find_study_lock(search_root='/kaggle/input'):
    locks=list(Path(search_root).rglob('study_lock.json'))
    if not locks: raise FileNotFoundError('Run 01_Audit_and_Lock, then attach its private output.')
    values={sha256_file(p):p for p in locks}
    if len(values)!=1: raise RuntimeError('Multiple different study locks attached.')
    lock=read_json(next(iter(values.values())))
    if lock['protocol_sha256']!=protocol_hash(): raise RuntimeError('Study protocol differs from the reviewed lock.')
    if lock.get('code_sha256') != code_hash(): raise RuntimeError('Code differs from the reviewed study lock.')
    if not lock.get('human_review_complete'): raise RuntimeError('Dataset review is not signed off.')
    return lock


def train_one(job,source,out,*,deadline,workers=2,pilot=False,pilot_images=2048,stop_file=None):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run=Path(out)/'medshift_runs'/job['job_id']; run.mkdir(parents=True,exist_ok=True)
    source_info=read_json(Path(source.iloc[0]['_root'])/'COMPLETE.json')
    signature=training_signature(job,source_info['manifest_sha256'],pilot)
    if validate_run(job,source_info,out,pilot): return True
    cfg=PROTOCOL.copy(); cfg.update(job); cfg.update(signature=signature,pilot=pilot,source_manifest_sha256=source_info['manifest_sha256'])
    train=nested_patient_subset(source[source.split=='train'],job['fraction'],job['seed'])
    tune=source[source.split=='tune'].copy()
    if pilot:
        train=train.head(pilot_images); tune=tune.head(512)
    for label in LABELS:
        if train[label].sum()<10 or (train[label]==0).sum()<10:
            if not pilot: raise RuntimeError(f'Insufficient positive/negative training labels for {label}. Revise protocol before external evaluation.')
    cfg.update(n_train_images=len(train),n_train_patients=train.patient_id.nunique(),n_tune_images=len(tune),
               n_total_development_images=len(train)+len(tune)+int((source.split=='calibration').sum()))
    seed_everything(job['seed'])
    model,weight_sha=make_model(job['model'],Path(out)/'initial_weights',pretrained=True)
    cfg['initial_weight_sha256']=weight_sha; cfg['parameters']=sum(p.numel() for p in model.parameters())
    cfg.update(code_sha256=code_hash(), software_version=SOFTWARE_VERSION,
               dataloader_workers=workers, cpu_threads=torch.get_num_threads(),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','unset'),
               gpu_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
               execution_engine='independent_trial_process')
    write_json(run/'run.json',cfg)
    model=model.to(device)
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':cfg['backbone_lr']},
                          {'params':list(model.head.parameters()),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start_epoch=0; best=float('inf'); best_epoch=-1; stale=0; history=[]
    last=run/'last.pt'
    if last.exists():
        # This is OUR trusted private checkpoint. Never load a stranger's pickle checkpoint here.
        ck=torch.load(last,map_location=device,weights_only=False)
        if ck['signature']!=signature: raise RuntimeError('Resume signature mismatch.')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1; best=ck['best']; best_epoch=ck['best_epoch']; stale=ck['stale']; history=ck['history']
    max_epochs=1 if pilot else cfg['epochs']
    valid_loader=make_loader(tune,batch=64,workers=workers)
    # Recover a session interrupted after the final checkpoint but before done.json.
    finished=(start_epoch>=max_epochs or (start_epoch>=cfg['min_epochs'] and stale>=cfg['patience']))
    for epoch in range(start_epoch, start_epoch if finished else max_epochs):
        # Stop before starting an epoch that is likely to outlive the notebook's own budget.
        estimate=history[-1]['wall_seconds'] if history else 600
        if pause_requested(deadline,stop_file) or time.monotonic()+max(120,estimate*1.2)>deadline:
            print('Safe pause before next epoch:',job['job_id'],epoch); break
        seed_everything(job['seed']*1000+epoch)
        fac=(epoch+1)/cfg['warmup_epochs'] if epoch<cfg['warmup_epochs'] else .5*(1+math.cos(math.pi*(epoch-cfg['warmup_epochs'])/max(1,cfg['epochs']-cfg['warmup_epochs'])))
        for pg,base in zip(opt.param_groups,[cfg['backbone_lr'],cfg['head_lr']]): pg['lr']=base*fac
        loader=make_loader(train,train=True,batch=cfg['microbatch'],workers=workers,seed=job['seed']*1000+epoch)
        model.train(); epoch_start=time.monotonic(); total=0.; batches=0
        opt.zero_grad(set_to_none=True); accumulation=cfg['effective_batch']//cfg['microbatch']
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        try:
            for bi,(x,y,_) in enumerate(loader):
                if pause_requested(deadline,stop_file):
                    raise PauseRequested('Pause during training; keep the previous completed epoch.')
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                # Scale the last partial accumulation window correctly.
                window_samples=min(cfg['effective_batch'],len(train)-(bi//accumulation)*accumulation*cfg['microbatch'])
                with autocast(device): loss=masked_bce(model(x),y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss.')
                scaler.scale(loss*(x.shape[0]/window_samples)).backward()
                if (bi+1)%accumulation==0 or bi+1==len(loader):
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                total+=float(loss.detach())*x.shape[0]; batches+=x.shape[0]
            val=validation_loss(model,valid_loader,device,deadline=deadline,stop_file=stop_file)
        except PauseRequested as exc:
            print('SAFE PAUSE:',job['job_id'],str(exc),flush=True)
            break
        if not math.isfinite(val): raise FloatingPointError('Non-finite tuning NLL.')
        improved=val<best-cfg['min_delta']
        if improved:
            best=val; best_epoch=epoch; stale=0
            atomic_torch_save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                               'signature':signature,'job':job,'best_epoch':epoch},run/'best.pt')
        else: stale+=1
        seconds=time.monotonic()-epoch_start
        history.append({'epoch':epoch,'train_bce':total/max(1,batches),'tune_nll':val,'wall_seconds':seconds,
                        'training_images_per_second_including_validation':len(train)/seconds,
                        'backbone_lr':opt.param_groups[0]['lr'],'device':str(device),
                        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unset'),
                        'peak_allocated_vram_mib':torch.cuda.max_memory_allocated(device)/(1024**2) if device.type=='cuda' else 0.})
        history_tmp=run/'history.csv.tmp'
        pd.DataFrame(history).to_csv(history_tmp,index=False)
        history_tmp.replace(run/'history.csv')
        atomic_torch_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                           'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,
                           'signature':signature},last)
        print(job['job_id'],f'epoch={epoch+1} tune_NLL={val:.5f} best={best:.5f} seconds={seconds:.1f}',flush=True)
        if epoch+1>=max_epochs or (epoch+1>=cfg['min_epochs'] and stale>=cfg['patience']):
            finished=True; break
    if finished:
        write_json(run/'done.json',{'job':job,'signature':signature,'best_epoch':best_epoch,'epochs_run':len(history),
                                   'best_tune_nll':best,'total_training_wall_seconds':sum(h['wall_seconds'] for h in history),
                                   'best_checkpoint_sha256':sha256_file(run/'best.pt'),'pilot':pilot})
        if last.exists(): last.unlink()  # retain best weights and logs; remove unnecessary optimizer storage
    del model,opt,scaler
    if device.type=='cuda': torch.cuda.empty_cache()
    return finished


def train_queue(*args,**kwargs):
    """Backward API alias; every notebook uses the dual-GPU dispatcher."""
    from .parallel import train_queue_parallel
    if 'workers' in kwargs: kwargs['workers_per_gpu']=kwargs.pop('workers')
    if 'require_lock' in kwargs and not kwargs.pop('require_lock'):
        raise ValueError('Full-study lock validation cannot be disabled.')
    return train_queue_parallel(*args,**kwargs)
