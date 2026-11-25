import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='DAG-free peak-to-fragment pseudo-label learning')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--archive',required=True);q.add_argument('--reference',required=True)
    q.add_argument('--output',required=True);q.add_argument('--workers',type=int,default=4);q.add_argument('--max-queries',type=int)
    q=sub.add_parser('audit');q.add_argument('--data',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('train');q.add_argument('--config',required=True);q.add_argument('--data',required=True);q.add_argument('--output',required=True)
    q.add_argument('--resume');q.add_argument('--initialize');q.add_argument('--limit',type=int);q.add_argument('--epochs',type=int);q.add_argument('--device',default='auto')
    q=sub.add_parser('evaluate');q.add_argument('--checkpoint',required=True);q.add_argument('--data',required=True);q.add_argument('--output',required=True)
    q.add_argument('--split',choices=['val','test'],default='test');q.add_argument('--sources',nargs='+',choices=['predicted','experimental'],default=['experimental'])
    q.add_argument('--limit',type=int);q.add_argument('--device',default='auto')
    q=sub.add_parser('infer');q.add_argument('--checkpoint',required=True);q.add_argument('--catalog',required=True)
    q.add_argument('--input',required=True);q.add_argument('--output',required=True);q.add_argument('--topk',type=int,default=10);q.add_argument('--device',default='auto')
    a=p.parse_args()
    if a.command=='prepare':
        from .data import prepare
        if a.workers<1:p.error('workers must be positive')
        result=prepare(a.archive,a.reference,a.output,a.workers,a.max_queries)
        print(json.dumps(result,indent=2),flush=True)
    elif a.command=='audit':
        from .audit import audit
        audit(a.data,a.output)
    elif a.command=='train':
        from .engine import train
        train(a.config,a.data,a.output,a.device,a.resume,a.initialize,a.limit,a.epochs)
    elif a.command=='evaluate':
        from .engine import evaluate
        evaluate(a.checkpoint,a.data,a.output,a.split,a.sources,a.device,a.limit)
    elif a.command=='infer':
        from .infer import infer
        infer(a.checkpoint,a.catalog,a.input,a.output,a.topk,a.device)


if __name__=='__main__':main()
