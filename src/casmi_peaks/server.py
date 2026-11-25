"""Reuse the project's uv environment and run a separately versioned peak pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from casmi_neural.server import exclusive_lock,ensure_prepared,run
from .data import digest,signature

ROOT=Path(__file__).resolve().parents[2]


def check(gpus,cpu):
    import numpy as np
    import torch,rdkit
    torch.from_numpy(np.zeros(1,dtype=np.float32))
    print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        rdkit=rdkit.__version__,visible_gpus=torch.cuda.device_count(),runtime_profile='FP32/math attention by default')),flush=True)
    if not cpu:
        if not torch.cuda.is_available() or torch.cuda.device_count()<gpus:raise ValueError('Not enough visible CUDA devices; check --gpus/CUDA_VISIBLE_DEVICES')
        for i in range(gpus):
            with torch.cuda.device(i):
                x=torch.ones(4,4,device=f'cuda:{i}',requires_grad=True);(x@x).sum().backward();torch.cuda.synchronize()


def ensure_data(args):
    from .data import prepare
    with exclusive_lock(args.data.parent/('.'+args.data.name+'.lock')):
        if args.data.exists() and any(args.data.iterdir()):
            meta=json.loads((args.data/'meta.json').read_text())
            if meta.get('schema_version')!=1 or meta.get('partial'):raise ValueError('Production launcher requires a complete v1 peak cache')
            for name,expected in meta['hashes'].items():
                if digest(args.data/name)!=expected:raise ValueError(f'Peak cache checksum mismatch: {name}')
            print(f'Reusing peak-label cache: {args.data}',flush=True);return meta
        args.reference.parent.mkdir(parents=True,exist_ok=True)
        ensure_prepared(args.archive,args.reference,args.workers)
        return prepare(args.archive,args.reference,args.data,args.workers)


def stage_state(path,config,meta,data):
    import torch
    path=Path(path)
    if not (path/'last.pt').exists():
        if (path/'best.pt').exists():raise ValueError('best.pt exists without last.pt')
        return 'new'
    ckpt=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    if ckpt['dataset_signature']!=signature(meta) or ckpt['config']!=config:
        raise ValueError('Existing run uses different data/configuration; select another --run-root')
    import hashlib
    index=[json.loads(line) for line in (Path(data)/'index.jsonl').read_text().splitlines()]
    selected=[[r for r in index if r['source'] in config['sources'] and r['split']==split and
               (split!='train' or r['annotated_peaks']>0)] for split in ['train','val']]
    subset=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
    if subset!=ckpt['subset_signature']:raise ValueError('Checkpoint used a limited/different subset; choose a new run root')
    if not (path/'best.pt').exists():raise ValueError('Missing best.pt')
    if ckpt['epoch']+1>=config['training']['epochs'] or ckpt['bad_epochs']>=config['training'].get('patience',12):return 'complete'
    return 'resume'


def launch(stage,args,meta):
    from dataclasses import asdict
    from .model import ModelConfig
    config_path=getattr(args,stage+'_config');config=json.loads(config_path.read_text())
    config['model']=asdict(ModelConfig(**config.get('model',{})))
    output=args.run_root/stage;state=stage_state(output,config,meta,args.data)
    if state=='complete':print(f'{stage}: complete, keeping {output / "best.pt"}',flush=True);return
    if args.cpu:command=[sys.executable,'-m','casmi_peaks','train']
    else:command=[sys.executable,'-m','torch.distributed.run','--nnodes=1',f'--nproc-per-node={args.gpus}',
            '--master-addr','127.0.0.1','--master-port',str(args.port),'-m','casmi_peaks','train']
    command+=['--config',config_path,'--data',args.data,'--output',output,'--device','cpu' if args.cpu else 'cuda']
    if state=='resume':command+=['--resume',output/'last.pt']
    elif stage=='finetune':
        checkpoint=args.run_root/'pretrain/best.pt'
        if not checkpoint.is_file():raise ValueError('Run pretrain before finetune')
        command+=['--initialize',checkpoint]
    run(command)


def main():
    p=argparse.ArgumentParser(description='DAG-free peak annotation on the existing AIMe/uv deployment')
    p.add_argument('command',choices=['check','prepare','audit','train','pretrain','finetune','evaluate','infer','status'])
    p.add_argument('--data',type=Path,default=ROOT/'data/prepared/peaks_v1')
    p.add_argument('--reference',type=Path,default=ROOT/'data/prepared/neural_v1')
    p.add_argument('--archive',type=Path,default=ROOT/'data/raw/casmi_top50_visualizer.zip')
    p.add_argument('--run-root',type=Path,default=ROOT/'runs/peaks_v1')
    p.add_argument('--pretrain-config',type=Path,default=ROOT/'configs/peaks/pretrain_h20.json')
    p.add_argument('--finetune-config',type=Path,default=ROOT/'configs/peaks/finetune_h20.json')
    p.add_argument('--gpus',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--port',type=int,default=29511);p.add_argument('--cpu',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);p.add_argument('--input',type=Path)
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--topk',type=int,default=10)
    args=p.parse_args()
    if args.gpus<1 or args.workers<1 or not 1<=args.port<=65535:p.error('Invalid GPU count, workers or port')
    if args.command=='check':check(args.gpus,args.cpu)
    elif args.command=='prepare':print(json.dumps(ensure_data(args),indent=2),flush=True)
    elif args.command=='audit':
        from .audit import audit
        audit(args.data,args.output or args.run_root/'data_audit.json')
    elif args.command in ['train','pretrain','finetune']:
        with exclusive_lock(args.run_root/'.peak-training.lock'):
            check(args.gpus,args.cpu);meta=ensure_data(args)
            for stage in ['pretrain','finetune'] if args.command=='train' else [args.command]:launch(stage,args,meta)
    elif args.command=='status':
        meta=json.loads((args.data/'meta.json').read_text())
        for stage in ['pretrain','finetune']:
            print(stage,stage_state(args.run_root/stage,json.loads(getattr(args,stage+'_config').read_text()),meta,args.data))
    elif args.command=='evaluate':
        from .engine import evaluate
        evaluate(args.checkpoint or args.run_root/'finetune/best.pt',args.data,
            args.output or args.run_root/'finetune/test.json',args.split,device_name='cpu' if args.cpu else 'auto')
    elif args.command=='infer':
        from .infer import infer
        if args.input is None:p.error('infer requires --input spectrum.json')
        infer(args.checkpoint or args.run_root/'finetune/best.pt',args.data,args.input,
            args.output or args.run_root/'peak_predictions.json',args.topk,'cpu' if args.cpu else 'auto')


if __name__=='__main__':main()
