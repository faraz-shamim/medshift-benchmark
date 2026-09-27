"""Fresh-interpreter task entry point. CUDA visibility is set by the parent first."""
from __future__ import annotations
import argparse
import os
import sys
import traceback
from pathlib import Path
from .common import read_json, write_json, code_hash, load_prepared


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args=parser.parse_args()
    req=read_json(args.request)
    result_path=Path(req['result_path'])
    try:
        if req['code_sha256']!=code_hash():
            raise RuntimeError('Worker source code differs from dispatcher.')
        # Imported only after the new interpreter received its CUDA device mask.
        import torch
        threads=int(req.get('cpu_threads',1))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('GPU worker must see exactly one GPU. Check Kaggle T4 x2 and CUDA masks.')
        torch.cuda.set_device(0)
        print(f'Worker PID={os.getpid()} physical CUDA mask={os.environ.get("CUDA_VISIBLE_DEVICES")} '
              f'local cuda:0={torch.cuda.get_device_name(0)} CPU threads={threads}',flush=True)
        if req['mode']=='train':
            from .train import train_one
            source,info=load_prepared(req['source_root'])
            done=train_one(req['job'],source,req['output'],deadline=req['deadline'],
                           workers=req['workers'],pilot=req['pilot'],stop_file=req['stop_file'])
        elif req['mode']=='evaluate':
            from .evaluate import inference_one
            done=inference_one(req['job'],req['run'],req['prepared_roots'],req['output'],
                               deadline=req['deadline'],workers=req['workers'],stop_file=req['stop_file'])
        else:
            raise ValueError(f'Unknown GPU task mode: {req["mode"]}')
        write_json(result_path,{'completed':bool(done),'pid':os.getpid(),
                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'gpu_name':torch.cuda.get_device_name(0),'code_sha256':code_hash()})
        return 0
    except BaseException as exc:
        write_json(result_path,{'completed':False,'error':f'{type(exc).__name__}: {exc}',
                               'traceback':traceback.format_exc(),'pid':os.getpid()})
        traceback.print_exc()
        return 1


if __name__=='__main__':
    sys.exit(main())
