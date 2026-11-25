"""Production V2 orchestration on the existing uv/H20 environment."""
import argparse
import json
from pathlib import Path
import sys
import torch
from casmi_neural.server import exclusive_lock,run
from casmi_peaks.server import check,ensure_data as ensure_legacy_data
from casmi_peaks.data import signature,digest,prepare
from . import IMPLEMENTATION_REVISION
from .engine import read_config
from .relations import prepare_relations,RelationStore

ROOT=Path(__file__).resolve().parents[2]


def ensure_data(args):
    if args.protocol=='legacy': return ensure_legacy_data(args)
    from casmi_neural.server import ensure_prepared
    from .protocol import prepare_query_reference
    with exclusive_lock(args.data.parent/('.'+args.data.name+'.lock')):
        if (args.data/'meta.json').exists():
            meta=json.loads((args.data/'meta.json').read_text())
            if meta['partial'] or meta['reference_signature']['config'].get('split')!='query_scaffold_intersection_v1':
                raise ValueError('Expected complete strict-query cache; do not reuse legacy cache for this protocol')
            for name,expected in meta['hashes'].items():
                if digest(args.data/name)!=expected: raise ValueError('Peak cache checksum mismatch')
            return meta
        ensure_prepared(args.archive,args.reference,args.workers)
        strict_reference=args.data.parent/(args.data.name+'_reference')
        prepare_query_reference(args.reference,strict_reference)
        return prepare(args.archive,strict_reference,args.data,args.workers)


def ensure_relations(args,meta):
    with exclusive_lock(args.relations.parent/('.'+args.relations.name+'.lock')):
        if (args.relations/'meta.json').exists():
            store=RelationStore(args.relations,meta)
            if store.meta['partial']: raise ValueError('Production requires complete relation data')
            result=store.meta; store.close(); print(f'Reusing DAG relation cache: {args.relations}',flush=True); return result
        reference=args.reference
        if args.protocol=='query-disjoint':
            reference=args.data.parent/(args.data.name+'_reference')
            if not (reference/'meta.json').exists():
                from .protocol import prepare_query_reference
                prepare_query_reference(args.reference,reference)
        result=prepare_relations(args.archive,reference,args.data,args.relations)
        if result['partial']: raise ValueError('Production requires complete relation data')
        return result


def stage_state(output,cfg,meta,relation_meta,data):
    import hashlib
    output=Path(output)
    if not (output/'last.pt').exists():
        if (output/'best.pt').exists(): raise ValueError('best.pt exists without last.pt')
        return 'new'
    ckpt=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
    if ckpt.get('implementation_revision')!=IMPLEMENTATION_REVISION:
        raise ValueError('Old implementation checkpoint; select a fresh audited run root')
    if ckpt.get('limited_training') or (ckpt.get('initialization') or {}).get('partial'):
        raise ValueError('Production stage has partial training/initialization lineage')
    if ckpt['dataset_signature']!=signature(meta) or ckpt['relation_signature']!=relation_meta or ckpt['config']!=cfg:
        raise ValueError('Existing V2 run has different data/relations/config; select another --run-root')
    index=[json.loads(line) for line in (Path(data)/'index.jsonl').read_text().splitlines()]
    subset=[[r for r in index if r['source'] in cfg['sources'] and r['split']==split and
             (split!='train' or r['annotated_peaks']>0)] for split in ('train','val')]
    subset.append([r for r in index if r['source'] in cfg['sources'] and r['split']=='train' and r['annotated_peaks']==0])
    if ckpt['subset_signature']!=hashlib.sha256(json.dumps(subset,sort_keys=True).encode()).hexdigest():
        raise ValueError('Checkpoint trained on a limited/different subset')
    if not (output/'best.pt').exists(): raise ValueError('Missing best checkpoint')
    return 'complete' if ckpt['epoch']+1>=cfg['training']['epochs'] or ckpt['bad_epochs']>=cfg['training'].get('patience',12) else 'resume'


def launch(stage,args,meta,relation_meta):
    config=getattr(args,stage+'_config'); cfg=read_config(config); output=args.run_root/stage
    state=stage_state(output,cfg,meta,relation_meta,args.data)
    if state=='complete': print(f'{stage}: complete, keeping {output / "best.pt"}',flush=True); return
    if args.cpu: command=[sys.executable,'-m','casmi_joint','train']
    else: command=[sys.executable,'-m','torch.distributed.run','--nnodes=1',f'--nproc-per-node={args.gpus}',
        '--master-addr','127.0.0.1','--master-port',str(args.port),'-m','casmi_joint','train']
    command+=['--config',config,'--data',args.data,'--relations',args.relations,'--output',output,'--device','cpu' if args.cpu else 'cuda']
    if state=='resume': command+=['--resume',output/'last.pt']
    elif stage=='finetune':
        checkpoint=args.run_root/'pretrain/best.pt'
        if not checkpoint.exists(): raise ValueError('Run V2 pretrain first')
        if stage_state(args.run_root/'pretrain',read_config(args.pretrain_config),meta,relation_meta,args.data)!='complete':
            raise ValueError('Production finetune requires completed full-data pretraining')
        command+=['--initialize',checkpoint]
    run(command)


def main():
    p=argparse.ArgumentParser(description='V2: peak-pair mass encoding, typed DAG relations, joint retrieval')
    p.add_argument('command',choices=['check','prepare','train','pretrain','finetune','calibrate','evaluate','infer','status','audit-relations','export-test'])
    p.add_argument('--protocol',choices=['query-disjoint','legacy'],default='query-disjoint')
    p.add_argument('--data',type=Path)
    p.add_argument('--relations',type=Path)
    p.add_argument('--reference',type=Path,default=ROOT/'data/prepared/neural_v1')
    p.add_argument('--archive',type=Path,default=ROOT/'data/raw/casmi_top50_visualizer.zip')
    p.add_argument('--run-root',type=Path)
    p.add_argument('--pretrain-config',type=Path,default=ROOT/'configs/peaks_v2/pretrain_h20.json')
    p.add_argument('--finetune-config',type=Path,default=ROOT/'configs/peaks_v2/finetune_h20.json')
    p.add_argument('--gpus',type=int,default=8); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--port',type=int,default=29512); p.add_argument('--cpu',action='store_true')
    p.add_argument('--checkpoint',type=Path); p.add_argument('--output',type=Path); p.add_argument('--input',type=Path)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--reference-metrics',type=Path)
    p.add_argument('--export-json',type=Path)
    p.add_argument('--source',choices=['predicted','experimental'],default='predicted',help='Spectrum source for evaluate/export-test; training is unchanged')
    p.add_argument('--split',choices=['val','test'],default='test'); p.add_argument('--topk',type=int,default=10)
    p.add_argument('--strength',type=float,help='Explicit ablation override; default uses validation-selected strength')
    a=p.parse_args()
    suffix='query' if a.protocol=='query-disjoint' else 'legacy_audit'
    a.data=a.data or ROOT/('data/prepared/peaks_query_v2' if suffix=='query' else 'data/prepared/peaks_v1')
    a.relations=a.relations or ROOT/f'data/prepared/peaks_v2_{suffix}_relations'
    a.run_root=a.run_root or ROOT/f'runs/peaks_v2_{suffix}'
    if a.gpus<1 or a.workers<1 or not 1<=a.port<=65535: p.error('Invalid GPU count, workers or port')
    checkpoint=a.checkpoint or a.run_root/'finetune/best.pt'; calibration=a.calibration or a.run_root/'calibration.json'
    device='cpu' if a.cpu else 'auto'
    if a.command=='check': check(a.gpus,a.cpu)
    elif a.command in ('prepare','train','pretrain','finetune'):
        with exclusive_lock(a.run_root/'.joint-training.lock'):
            if a.command!='prepare': check(a.gpus,a.cpu)
            meta=ensure_data(a); relation_meta=ensure_relations(a,meta)
            if a.command=='prepare': print(json.dumps(relation_meta,indent=2),flush=True); return
            for stage in ('pretrain','finetune') if a.command=='train' else (a.command,): launch(stage,a,meta,relation_meta)
            if a.command in ('train','finetune'):
                from .evaluate import calibrate
                calibrate(a.run_root/'finetune/best.pt',a.data,calibration,device)
                from .audit import audit_relations
                audit_relations(a.run_root/'finetune/best.pt',a.data,a.relations,a.run_root/'relation_validation.json',device)
    elif a.command=='status':
        meta=json.loads((a.data/'meta.json').read_text()); rmeta=json.loads((a.relations/'meta.json').read_text())
        for stage in ('pretrain','finetune'): print(stage,stage_state(a.run_root/stage,read_config(getattr(a,stage+'_config')),meta,rmeta,a.data))
    elif a.command=='calibrate':
        from .evaluate import calibrate
        calibrate(checkpoint,a.data,a.output or calibration,device)
    elif a.command=='audit-relations':
        from .audit import audit_relations
        audit_relations(checkpoint,a.data,a.relations,a.output or a.run_root/'relation_validation.json',device,'val')
    elif a.command=='evaluate':
        from .evaluate import evaluate
        evaluate(checkpoint,a.data,a.output or a.run_root/f'finetune/{a.split}_{a.source}.json',a.split,device,
                 calibration if calibration.exists() else None,a.strength,source=a.source)
    elif a.command=='infer':
        from .infer import infer
        if a.input is None: p.error('infer requires --input')
        infer(checkpoint,a.data,a.input,a.output or a.run_root/'peak_predictions.json',a.topk,device,
              calibration if calibration.exists() else None,a.strength)
    elif a.command=='export-test':
        if a.strength is not None:p.error('export-test uses the matched evaluation calibration; --strength is not supported here')
        from .export import export_test
        reference=a.reference_metrics or (a.run_root/f'finetune/{a.split}_{a.source}.json')
        if reference is not None and not reference.exists():
            raise ValueError('Run evaluate first, or supply --reference-metrics with the matching evaluation JSON')
        export_test(a.data,a.archive,a.output or a.run_root/f'test-atlas-{a.source}.html',checkpoint,
            calibration if calibration.exists() else None,reference,a.topk,a.split,device,json_output=a.export_json,source=a.source)


if __name__=='__main__': main()
