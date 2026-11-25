#!/usr/bin/env python3
"""Generate explicit paired V2 ablations; does not launch concurrent GPU jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def generate(output,seeds=(7,17,27)):
    root=Path(__file__).resolve().parents[1]; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    base={stage:json.loads((root/f'configs/peaks_v2/{stage}_h20.json').read_text()) for stage in ('pretrain','finetune')}
    cases={
        'full':dict(use_pair_bias=True,peak_only=False,relations=True,joint=True),
        'unary_no_relations':dict(use_pair_bias=True,peak_only=False,relations=False,joint=False),
        'unary_with_relations':dict(use_pair_bias=True,peak_only=False,relations=True,joint=False),
        'no_pair_bias':dict(use_pair_bias=False,peak_only=False,relations=True,joint=True),
        'peak_only_unary':dict(use_pair_bias=True,peak_only=True,relations=False,joint=False),
    }
    manifest=[]
    for case,options in cases.items():
        for seed in seeds:
            row=dict(case=case,seed=int(seed),protocol='query-disjoint',stages={})
            for stage,original in base.items():
                cfg=copy.deepcopy(original); cfg['seed']=int(seed)
                cfg['model'].update(use_pair_bias=options['use_pair_bias'],peak_only=options['peak_only'])
                if not options['relations']: cfg['training']['relation_weight']=0.
                if not options['joint']: cfg['joint']['strength']=0.
                if not options['joint']: cfg['calibration_strengths']=[0.]
                path=output/f'{case}_s{seed}_{stage}.json'
                payload=json.dumps(cfg,indent=2,allow_nan=False)+'\n'
                if path.exists() and path.read_text()!=payload: raise ValueError(f'Refusing to overwrite different config: {path}')
                path.write_text(payload); row['stages'][stage]=dict(path=str(path),sha256=hashlib.sha256(payload.encode()).hexdigest())
            row['run_root']=f'runs/experiments/{case}_s{seed}'
            # Zero-joint ablations must remain zero at evaluation too; generic
            # validation calibration is otherwise allowed to turn it back on.
            row['evaluation_strength_override']=None if options['joint'] else 0.
            manifest.append(row)
    (output/'manifest.json').write_text(json.dumps(dict(cases=cases,runs=manifest,
        note='Use fresh per-case pretraining. --strength 0 is mandatory when evaluating zero-joint ablations; calibration only uses validation.'),indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27]); args=p.parse_args()
    print(json.dumps(dict(configured_runs=len(generate(args.output,args.seeds)),manifest=str(args.output/'manifest.json'))))
