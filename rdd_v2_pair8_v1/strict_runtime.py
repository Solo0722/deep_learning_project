"""Strict execution settings and descriptive resource records; no policy changes."""
import json
import os
from pathlib import Path
import statistics
import time
import uuid

import torch
import torch.utils.deterministic

import baseline as b
import diagnostic as d
import observer


def state():
    return {'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'fill_uninitialized_memory':torch.utils.deterministic.fill_uninitialized_memory}


def check(cfg):
    d.require(state()==cfg['strict_settings'],'Strict execution settings changed')


def configure(cfg, fresh_cuda=True):
    d.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8',
              'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    if fresh_cuda:d.require(not torch.cuda.is_initialized(),'CUDA initialized before strict settings')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.utils.deterministic.fill_uninitialized_memory=True
    check(cfg)


def memory(device):
    if not str(device).startswith('cuda'):return {}
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'device_total_bytes':torch.cuda.get_device_properties(0).total_memory}


def reset(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()


class Window:
    """Measure at most 128 normal update calls per fit invocation, without hashing."""
    def __init__(self,cfg,device,out,progress):
        self.cfg,self.device,self.out=cfg,device,Path(out)
        self.start_step=progress['step'];self.seconds=[];self.saved=False
        self.cap=cfg['resource_checks']['first_window_updates']

    def begin(self):
        if len(self.seconds)>=self.cap:return
        if not self.seconds:reset(self.device)
        elif str(self.device).startswith('cuda'):torch.cuda.synchronize()
        self.start=time.monotonic()

    def end(self):
        if len(self.seconds)>=self.cap:return
        if str(self.device).startswith('cuda'):torch.cuda.synchronize()
        self.seconds.append(time.monotonic()-self.start)
        if len(self.seconds)==self.cap:self.finish()

    def finish(self):
        if self.saved or not self.seconds:return
        folder=self.out/'resources';folder.mkdir(exist_ok=True)
        result={'start_step':self.start_step,'updates':len(self.seconds),'seconds':self.seconds,
            'median_update_seconds':statistics.median(self.seconds),'sum_update_seconds':sum(self.seconds),
            'scope':'Actual baseline.update wall time with synchronization; excludes data loading, probes, validation and checkpoint writes. Gate observers execute outside the measured update.',
            'strict_settings':state(),**memory(self.device)}
        b.write_json(folder/f'training_window_{self.start_step:08d}_{uuid.uuid4().hex[:12]}.json',result)
        self.saved=True
        print(f"RESOURCE training_updates={len(self.seconds)} median_seconds={result['median_update_seconds']:.3f} peak_reserved_GiB={result.get('peak_reserved_bytes',0)/2**30:.3f}",flush=True)


def collect(model,optimizer,root,rows,plan,cfg,epoch,device,out,should_stop):
    check(cfg);reset(device);start=time.monotonic()
    limit=cfg['resource_checks']['maximum_probe_seconds']
    limited=lambda:should_stop() or time.monotonic()-start>=limit
    result=None;outcome='failed'
    try:
        result=observer.collect(model,optimizer,root,rows,plan,cfg,epoch,device,out,limited)
        elapsed=time.monotonic()-start
        if elapsed>=limit:
            outcome='resource_limit'
            raise RuntimeError('Controller probe reached its declared time limit; inspect records before continuing')
        outcome='complete' if result is not None else 'budget_stopped'
        check(cfg)
        return result
    finally:
        folder=Path(out)/'resources';folder.mkdir(exist_ok=True)
        record={'kind':'controller_probe','epoch':epoch,'images':len(plan),'status':outcome,
            'elapsed_seconds':time.monotonic()-start,'maximum_seconds':limit,
            'strict_settings':state(),**memory(device)}
        b.write_json(folder/f'probe_epoch_{epoch:03d}_{uuid.uuid4().hex[:12]}.json',record)
        print(f"RESOURCE probe_epoch={epoch} status={outcome} seconds={record['elapsed_seconds']:.2f} peak_reserved_GiB={record.get('peak_reserved_bytes',0)/2**30:.3f}",flush=True)
