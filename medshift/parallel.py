"""Independent experiment scheduling: one isolated Python process per GPU slot.

This is NOT DDP/DataParallel. No model, gradients, minibatch or optimizer is shared.
The parent alone dispatches work and writes global status; workers own disjoint run
folders. A new process per experiment releases all CUDA allocations at job exit.
"""
from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

from .common import (SOFTWARE_VERSION, PROTOCOL, MODELS, SEEDS, FRACTIONS, DOMAINS,
                     jobs, job_id, load_prepared, assert_patient_disjoint, read_json,
                     write_json, environment, code_hash, protocol_hash, check_storage)

EXECUTION_PLAN = {
    'strategy': 'independent_trials_one_process_per_gpu',
    'max_parallel_models': 2,
    'dataloader_workers_per_model': 1,
    'cpu_threads_per_model': 1,
    'per_model_microbatch': 32,
    'per_model_effective_batch': 64,
    'distributed_gradient_sync': False,
}


def validate_gpu_ids(gpu_ids=(0, 1), require_two=True):
    """Return CUDA visibility tokens mapped from the parent's logical GPU indices."""
    import torch
    ids = list(gpu_ids)
    if not ids or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in ids):
        raise ValueError('GPU_IDS must be a nonempty list of nonnegative integers.')
    if len(ids) != len(set(ids)):
        raise ValueError('Each GPU may appear only once in GPU_IDS.')
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or any(x >= count for x in ids):
        raise RuntimeError(f'Requested GPUs {ids}; CUDA exposes {count}. Select Kaggle GPU T4 x2.')
    if require_two and len(ids) != 2:
        raise RuntimeError('This release expects GPU_IDS=[0,1]. Do not silently fall back to one GPU.')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    mapping = [x.strip() for x in inherited.split(',')] if inherited else []
    if mapping and len(mapping) < count:
        raise RuntimeError('CUDA_VISIBLE_DEVICES is inconsistent. Start a clean Kaggle session.')
    tokens = {x: mapping[x] if mapping else str(x) for x in ids}
    for x in ids:
        print(f'GPU slot {x}: {torch.cuda.get_device_name(x)}; CUDA_VISIBLE_DEVICES={tokens[x]}', flush=True)
    return tokens


def child_environment(gpu_token, threads=1):
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer.')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_token)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONHASHSEED'] = '0'
    env['MEDSHIFT_CPU_THREADS'] = str(threads)
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Kaggle provides four CPU cores shared by both GPUs. Avoid oversubscription.
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[key] = str(threads)
    root = str(Path(__file__).resolve().parent.parent)
    prior = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = root + (os.pathsep + prior if prior else '')
    return env


def _event(path, **event):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 **event}, sort_keys=True) + '\n')


def _tail(entry):
    """Stream a file instead of using PIPE: no unconsumed-pipe deadlock."""
    with open(entry['log_path'], 'r', encoding='utf-8', errors='replace') as stream:
        stream.seek(entry.get('log_offset', 0))
        text = stream.read()
        entry['log_offset'] = stream.tell()
    if text:
        for line in text.splitlines():
            print(f'[GPU {entry["gpu"]} | {entry["task_id"]}] {line}', flush=True)


def _gpu_snapshot(path):
    query = ['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits']
    try:
        text = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return
    new = not Path(path).exists()
    with open(path, 'a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(['utc', 'gpu_index', 'gpu_uuid', 'name', 'utilization_percent',
                             'memory_used_mib', 'memory_total_mib'])
        for row in csv.reader(text.splitlines(), skipinitialspace=True):
            writer.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), *row])


def _kill_group(process, sig):
    try:
        if os.name == 'posix':
            os.killpg(process.pid, sig)
        else:
            process.terminate() if sig == signal.SIGTERM else process.kill()
    except ProcessLookupError:
        pass


def run_task_pool(tasks, gpu_tokens, session_dir, *, deadline, threads=1,
                  worker_module='medshift.gpu_worker', progress_callback=None,
                  monitor=True, poll_seconds=0.5, launch_margin_seconds=180,
                  shutdown_grace_seconds=90):
    """Dispatch distinct tasks at most once per session, one task per GPU slot.

    Worker requests/results live in private session output. A paused task is not
    retried in the same session. Nonzero exits are surfaced, never treated as done.
    The deadline is one shared monotonic-clock wall-time budget, not GPU-hours.
    """
    session_dir = Path(session_dir); session_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(tasks)
    ids = [t['task_id'] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task IDs would race on checkpoints.')
    if not gpu_tokens or len(set(gpu_tokens.values())) != len(gpu_tokens):
        raise ValueError('GPU slots must bind to distinct CUDA devices.')
    if not math.isfinite(deadline):
        raise ValueError('Finite monotonic deadline required.')
    todo = deque(tasks); active = {}; blocked = set(); results = {}; failure = None
    stop_file = session_dir/'STOP_REQUESTED'
    events = session_dir/'events.jsonl'
    last_monitor = 0.; stopping_since = None
    # Every session has a unique directory; do not reuse its stop marker.
    if stop_file.exists():
        raise RuntimeError('Session directory already has a stop marker. Use a new session directory.')
    try:
        while todo or active:
            now = time.monotonic()
            if now >= deadline and stopping_since is None:
                stop_file.touch(); stopping_since = now
                _event(events, event='deadline_reached')
                print('Shared deadline reached: keeping completed epochs/contexts and stopping workers.', flush=True)
            if failure is None and stopping_since is None:
                for gpu, token in gpu_tokens.items():
                    if not todo or now + launch_margin_seconds >= deadline:
                        break
                    if gpu in active or gpu in blocked:
                        continue
                    task = todo.popleft(); task_id = task['task_id']
                    # IDs generated by the package contain no path separators.
                    if '/' in task_id or '\\' in task_id or task_id in {'.', '..'}:
                        raise ValueError('Unsafe task ID.')
                    request_path = session_dir/f'{task_id}.request.json'
                    result_path = session_dir/f'{task_id}.result.json'
                    payload = {**task, 'deadline': deadline, 'stop_file': str(stop_file),
                               'result_path': str(result_path), 'gpu_slot': gpu,
                               'cuda_token': str(token), 'cpu_threads': threads,
                               'code_sha256': code_hash(), 'software_version': SOFTWARE_VERSION}
                    write_json(request_path, payload)
                    log_path = session_dir/f'{task_id}.log'
                    log_handle = open(log_path, 'ab', buffering=0)
                    try:
                        proc = subprocess.Popen([sys.executable, '-u', '-m', worker_module,
                                                 '--request', str(request_path)],
                                                stdout=log_handle, stderr=subprocess.STDOUT,
                                                env=child_environment(token, threads),
                                                start_new_session=(os.name == 'posix'))
                    except BaseException:
                        log_handle.close(); raise
                    active[gpu] = {'process': proc, 'log_handle': log_handle, 'log_path': log_path,
                                   'result_path': result_path, 'task_id': task_id,
                                   'gpu': gpu, 'started': time.monotonic()}
                    _event(events, event='started', gpu_slot=gpu, cuda_token=str(token),
                           task_id=task_id, pid=proc.pid)
                    print(f'[START GPU {gpu}] {task_id}; isolated PID={proc.pid}', flush=True)
            for gpu, entry in list(active.items()):
                _tail(entry)
                rc = entry['process'].poll()
                if rc is None:
                    continue
                entry['log_handle'].close(); _tail(entry)
                task_id = entry['task_id']
                if rc != 0 or not entry['result_path'].exists():
                    result = {'completed': False, 'error': f'Worker exit={rc}; inspect {entry["log_path"]}'}
                    failure = failure or result['error']
                else:
                    result = read_json(entry['result_path'])
                    if result.get('error'):
                        failure = failure or str(result['error'])
                results[task_id] = {**result, 'gpu_slot': gpu,
                                    'process_wall_seconds': time.monotonic()-entry['started']}
                _event(events, event='finished', task_id=task_id, gpu_slot=gpu,
                       exit_code=rc, completed=bool(result.get('completed')))
                del active[gpu]
                if not result.get('completed'):
                    blocked.add(gpu)
                if progress_callback is not None:
                    progress_callback()
                if failure is not None and stopping_since is None:
                    stop_file.touch(); stopping_since=time.monotonic()
                    print('Worker failed; asking the other GPU to pause safely.', flush=True)
            now = time.monotonic()
            if now-last_monitor >= 60:
                if monitor: _gpu_snapshot(session_dir/'gpu_monitor.csv')
                _event(events, event='heartbeat', active={str(g):e['task_id'] for g,e in active.items()},
                       completed_in_session=sum(bool(r.get('completed')) for r in results.values()),
                       queued=len(todo), seconds_remaining=max(0,deadline-now))
                print(f'[QUEUE] active={ {g:e["task_id"] for g,e in active.items()} }; '
                      f'pending={len(todo)}; session minutes left={max(0,deadline-now)/60:.1f}', flush=True)
                last_monitor=now
            if stopping_since is not None and now-stopping_since > shutdown_grace_seconds and active:
                for entry in active.values(): _kill_group(entry['process'], signal.SIGTERM)
                time.sleep(min(1., poll_seconds))
                for entry in active.values():
                    if entry['process'].poll() is None: _kill_group(entry['process'], signal.SIGKILL)
                failure = failure or ('A worker did not stop within the grace period. '
                                      'Resume only completed epoch/context checkpoints.')
            if not active and (not todo or len(blocked)==len(gpu_tokens) or failure is not None
                               or time.monotonic()+launch_margin_seconds >= deadline):
                break
            time.sleep(poll_seconds)
    except BaseException as exc:
        failure = failure or f'{type(exc).__name__}: {exc}'
        stop_file.touch(exist_ok=True)
        raise
    finally:
        # Handles KeyboardInterrupt and exceptions in the parent; do not orphan GPU loaders.
        if active:
            stop_file.touch(exist_ok=True)
            end=time.monotonic()+shutdown_grace_seconds
            while any(e['process'].poll() is None for e in active.values()) and time.monotonic()<end:
                for entry in active.values(): _tail(entry)
                time.sleep(poll_seconds)
            for entry in active.values():
                if entry['process'].poll() is None: _kill_group(entry['process'], signal.SIGTERM)
            for entry in active.values():
                try: entry['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_group(entry['process'], signal.SIGKILL); entry['process'].wait(timeout=10)
                entry['log_handle'].close(); _tail(entry)
        write_json(session_dir/'session_result.json', {'results':results,
                   'not_dispatched':[t['task_id'] for t in todo], 'error':failure,
                   'software_version':SOFTWARE_VERSION, 'code_sha256':code_hash()})
    if failure is not None:
        raise RuntimeError(f'Dual-GPU session stopped: {failure}')
    return results


def _session_root(out):
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    return Path(out)/'sessions'/name


def _check_budget(max_hours):
    if not isinstance(max_hours, (int,float)) or not math.isfinite(max_hours) or not 0 < max_hours <= 11:
        raise ValueError('MAX_HOURS must be >0 and <=11. Kaggle session/quota limits still apply.')


def _training_status(out, todo, *, pilot):
    import pandas as pd
    rows=[]
    for job in todo:
        run=Path(out)/'medshift_runs'/job['job_id']
        history=pd.read_csv(run/'history.csv') if (run/'history.csv').exists() else pd.DataFrame()
        rows.append({**job, 'completed':(run/'done.json').exists(), 'saved_epochs':len(history),
                     'last_cuda_device':str(history.cuda_visible_devices.iloc[-1])
                      if len(history) and 'cuda_visible_devices' in history else ''})
    frame=pd.DataFrame(rows); frame.to_csv(Path(out)/'queue_status.csv', index=False)
    write_json(Path(out)/'queue_summary.json', {'completed':int(frame.completed.sum()),
               'expected':len(todo), 'pilot':pilot, 'strategy':EXECUTION_PLAN['strategy'],
               'software_version':SOFTWARE_VERSION})
    return frame


def _pilot_report(out, source, results, session):
    import pandas as pd
    measurements=[]
    for model in MODELS:
        jid=job_id(model,.1,SEEDS[0]); run=Path(out)/'medshift_runs'/jid
        if not (run/'done.json').exists(): continue
        cfg=read_json(run/'run.json'); hist=pd.read_csv(run/'history.csv')
        measurements.append({'model':model, 'training_images':cfg['n_train_images'],
                             'tuning_images':cfg['n_tune_images'],
                             'epoch_seconds':float(hist.wall_seconds.iloc[0]),
                             'peak_allocated_vram_mib':float(hist.peak_allocated_vram_mib.max()),
                             'cuda_visible_devices':cfg['cuda_visible_devices']})
    write_json(Path(out)/'pilot_budget.json', {'models_tested':measurements,
               'completed':len(measurements), 'expected':len(MODELS),
               'full_training_source_images':int((source.split=='train').sum()),
               'session_dir':str(session),
               'caution':'Three tiny one-epoch engineering trials, not study results or a validated '
                         'runtime forecast. Shared CPU/I/O and full-cohort validation change throughput. '
                         'Measure the first full session; do not assume a 2x speedup.'})


def train_queue_parallel(source_root, output='/kaggle/working/training', *, max_hours=11.,
                         workers_per_gpu=1, threads_per_gpu=1, gpu_ids=(0,1),
                         pilot=False, restore=True, search_root='/kaggle/input'):
    import pandas as pd
    from .train import find_study_lock, copy_prior_runs, validate_run
    _check_budget(max_hours)
    if not isinstance(workers_per_gpu,int) or workers_per_gpu<0:
        raise ValueError('WORKERS_PER_GPU must be a nonnegative integer.')
    deadline=time.monotonic()+max_hours*3600  # includes restoring files and worker setup
    tokens=validate_gpu_ids(gpu_ids)
    source,info=load_prepared(source_root); assert_patient_disjoint(source)
    if info.get('test_mode') and not pilot:
        raise RuntimeError('Synthetic/test preparation cannot enter manuscript training.')
    out=Path(output)/('pilot' if pilot else 'full'); out.mkdir(parents=True,exist_ok=True)
    if not pilot:
        lock=find_study_lock(search_root)
        if lock['datasets']['nih']!=info['manifest_sha256']:
            raise RuntimeError('NIH manifest differs from the reviewed lock.')
        if lock.get('execution_plan')!=EXECUTION_PLAN:
            raise RuntimeError('Run the UPDATED notebook 01 and attach its new lock, not the old lock.')
        if workers_per_gpu!=lock['execution_plan']['dataloader_workers_per_model'] or \
                threads_per_gpu!=lock['execution_plan']['cpu_threads_per_model']:
            raise RuntimeError('Worker/thread settings differ from the reviewed execution plan.')
        write_json(out/'study_lock.json',lock)
    if restore and not pilot:
        # Refuse old pilots before copying their weights/runs into manuscript output.
        for marker in Path(search_root).rglob('medshift_runs/*/done.json'):
            if read_json(marker).get('pilot'):
                raise RuntimeError('Remove attached pilot output before a full run.')
        copy_prior_runs(out,search_root)
    todo=jobs() if not pilot else [j for j in jobs() if j['fraction']==.1 and j['seed']==SEEDS[0]]
    expected_ids={j['job_id'] for j in todo}
    unexpected=[p.parent.name for p in (out/'medshift_runs').glob('*/run.json') if p.parent.name not in expected_ids]
    if unexpected: raise RuntimeError(f'Unexpected prior run IDs: {unexpected}')
    pending=[j for j in todo if not validate_run(j,info,out,pilot)]
    marker=out/'TRAINING_COMPLETE.json'
    if marker.exists(): marker.unlink()
    session=_session_root(out); session.mkdir(parents=True)
    environment(session/'environment.json')
    write_json(out/'protocol.json',PROTOCOL)
    write_json(out/'execution_plan.json',EXECUTION_PLAN)
    tasks=[{'task_id':j['job_id'], 'mode':'train', 'job':j, 'source_root':str(source_root),
            'output':str(out), 'pilot':pilot, 'workers':workers_per_gpu} for j in pending]
    print(f'{len(todo)-len(pending)}/{len(todo)} already complete; '
          f'{len(pending)} queued on {len(tokens)} GPUs. pilot={pilot}',flush=True)
    results={}
    try:
        results=run_task_pool(tasks,tokens,session,deadline=deadline,threads=threads_per_gpu,
                             progress_callback=lambda:_training_status(out,todo,pilot=pilot))
    finally:
        status=_training_status(out,todo,pilot=pilot)
        if pilot: _pilot_report(out,source,results,session)
        print(status.to_string(index=False),flush=True)
    complete=all(validate_run(j,info,out,pilot) for j in todo)
    if complete and not pilot:
        write_json(marker,{'jobs':len(todo),'protocol_sha256':protocol_hash(),
                           'code_sha256':code_hash(),'software_version':SOFTWARE_VERSION})
    check_storage('/kaggle/working' if Path('/kaggle/working').exists() else out,reserve_gb=.5)
    print('Save PRIVATE. Resume full runs with ONLY the latest cumulative full output plus NIH/new lock.',flush=True)
    return out


def inference_queue_parallel(prepared_roots, output='/kaggle/working/evaluation', *, max_hours=11.,
                             workers_per_gpu=1, threads_per_gpu=1, gpu_ids=(0,1),
                             search_root='/kaggle/input', external_evaluation_authorized=False):
    from .evaluate import (prepare_evaluation, evaluation_status, job_predictions_complete)
    _check_budget(max_hours)
    if not external_evaluation_authorized:
        raise PermissionError('Set EXTERNAL_EVALUATION_AUTHORIZED only after protocol and 36 models are frozen.')
    deadline=time.monotonic()+max_hours*3600
    tokens=validate_gpu_ids(gpu_ids)
    out,lock,runs,contexts,infos=prepare_evaluation(prepared_roots,output,search_root)
    if lock.get('execution_plan')!=EXECUTION_PLAN:
        raise RuntimeError('The updated execution-plan lock is required.')
    session=_session_root(out); session.mkdir(parents=True)
    environment(session/'environment.json')
    tasks=[]
    for job in jobs():
        if not job_predictions_complete(job,runs[job['job_id']],contexts,lock,out):
            tasks.append({'task_id':job['job_id'], 'mode':'evaluate', 'job':job,
                          'prepared_roots':{k:str(v) for k,v in prepared_roots.items()},
                          'run':str(runs[job['job_id']]),'output':str(out),'workers':workers_per_gpu})
    marker=out/'INFERENCE_COMPLETE.json'
    if marker.exists(): marker.unlink()
    try:
        run_task_pool(tasks,tokens,session,deadline=deadline,threads=threads_per_gpu)
    finally:
        status,hashes,metadata_hashes=evaluation_status(out,runs,contexts,lock)
        print(f'{int(status.complete.sum())}/{len(status)} prediction files complete.',flush=True)
    if bool(status.complete.all()):
        write_json(marker,{'prediction_files':len(status),'protocol_sha256':protocol_hash(),
                           'dataset_hashes':lock['datasets'],'code_sha256':code_hash(),
                           'prediction_hashes':hashes,'metadata_hashes':metadata_hashes,
                           'software_version':SOFTWARE_VERSION})
    check_storage('/kaggle/working' if Path('/kaggle/working').exists() else out,reserve_gb=.5)
    print('Save PRIVATE; keep completed contexts. A paused context is recomputed on resume.',flush=True)
    return out
