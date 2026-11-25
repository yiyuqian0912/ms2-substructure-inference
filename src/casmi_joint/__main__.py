import argparse
import json


def main():
    p=argparse.ArgumentParser(description='Mass-aware spectrum encoder and train-DAG relation-constrained peak retrieval')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare-relations')
    for key in ('archive','reference','data','output'): q.add_argument('--'+key,required=True)
    q=sub.add_parser('train')
    for key in ('config','data','relations','output'): q.add_argument('--'+key,required=True)
    q.add_argument('--resume'); q.add_argument('--initialize'); q.add_argument('--limit',type=int); q.add_argument('--device',default='auto')
    q=sub.add_parser('audit-relations')
    for key in ('checkpoint','data','relations','output'): q.add_argument('--'+key,required=True)
    q.add_argument('--device',default='auto'); q.add_argument('--split',choices=['train','val'],default='val')
    q=sub.add_parser('export-test')
    for key in ('data','archive','output'):q.add_argument('--'+key,required=True)
    q.add_argument('--checkpoint');q.add_argument('--calibration');q.add_argument('--reference-metrics')
    q.add_argument('--topk',type=int,default=10);q.add_argument('--split',choices=['val','test'],default='test')
    q.add_argument('--device',default='auto');q.add_argument('--teacher-only',action='store_true');q.add_argument('--limit',type=int)
    q.add_argument('--json-output')
    q.add_argument('--source',choices=['predicted','experimental'],default='predicted')
    for name in ('evaluate','calibrate','infer'):
        q=sub.add_parser(name)
        for key in ('checkpoint','data','output'): q.add_argument('--'+key,required=True)
        q.add_argument('--device',default='auto')
        if name!='calibrate': q.add_argument('--calibration'); q.add_argument('--strength',type=float)
        if name=='evaluate':
            q.add_argument('--split',choices=['val','test'],default='test'); q.add_argument('--limit',type=int)
            q.add_argument('--source',choices=['predicted','experimental'],default='predicted')
        if name=='infer': q.add_argument('--input',required=True); q.add_argument('--topk',type=int,default=10)
    a=p.parse_args(); args=vars(a); command=args.pop('command')
    if command=='prepare-relations':
        from .relations import prepare_relations
        print(json.dumps(prepare_relations(**args),indent=2),flush=True)
    elif command=='train':
        from .engine import train
        args['device_name']=args.pop('device'); train(**args)
    elif command=='evaluate':
        from .evaluate import evaluate
        args['device_name']=args.pop('device'); evaluate(**args)
    elif command=='calibrate':
        from .evaluate import calibrate
        args['device_name']=args.pop('device'); calibrate(**args)
    elif command=='audit-relations':
        from .audit import audit_relations
        args['device_name']=args.pop('device'); audit_relations(**args)
    elif command=='infer':
        from .infer import infer
        args['device_name']=args.pop('device'); args['input_path']=args.pop('input'); infer(**args)
    elif command=='export-test':
        from .export import export_test
        args['device_name']=args.pop('device'); export_test(**args)


if __name__=='__main__': main()

