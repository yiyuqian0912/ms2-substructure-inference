import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Unlabeled spectrum-conditioned DAG decomposition')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--max-queries',type=int)
    p.add_argument('--split',choices=['scaffold','identity'],default='scaffold');p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--tolerance-da',type=float,default=.007);p.add_argument('--tolerance-ppm',type=float,default=5.)
    p.add_argument('--intensity-power',type=float,default=.5)
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--device',default='auto')
    p.add_argument('--backend',choices=['nccl','gloo']);p.add_argument('--resume');p.add_argument('--initialize')
    p.add_argument('--epochs',type=int);p.add_argument('--data');p.add_argument('--output');p.add_argument('--workers',type=int)
    p.add_argument('--limit',type=int)
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--sources',nargs='+',choices=['predicted','experimental'])
    p.add_argument('--device',default='auto');p.add_argument('--backend',choices=['nccl','gloo'])
    p.add_argument('--output');p.add_argument('--workers',type=int,default=0);p.add_argument('--limit',type=int)
    p=sub.add_parser('infer');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data');p.add_argument('--sample-id',type=int);p.add_argument('--entry-json');p.add_argument('--entry-index',type=int,default=0)
    p.add_argument('--source',choices=['predicted','experimental'],default='experimental');p.add_argument('--device',default='cpu')
    p.add_argument('--spectrum-json',help='Override the spectrum in --entry-json with an independent [mz,intensity] list')
    args=parser.parse_args();kw=vars(args);command=kw.pop('command')
    if command=='prepare':
        from .data import PrepareConfig,prepare
        if args.tolerance_da<0 or args.tolerance_ppm<=0 or not 0<args.intensity_power<=1:
            parser.error('Require tolerance_da>=0, tolerance_ppm>0 and 0<intensity_power<=1')
        cfg=PrepareConfig(args.tolerance_da,args.tolerance_ppm,args.intensity_power,args.seed,args.split)
        prepare(args.archive,args.output,cfg,args.workers,args.max_queries)
    elif command=='train':
        from .train import train
        train(kw.pop('config'),device_name=kw.pop('device'),epochs_override=kw.pop('epochs'),
              data_override=kw.pop('data'),output_override=kw.pop('output'),workers_override=kw.pop('workers'),**kw)
    elif command=='evaluate':
        from .train import evaluate
        evaluate(device_name=kw.pop('device'),**kw)
    elif command=='infer':
        from .infer import infer
        infer(device_name=kw.pop('device'),**kw)


if __name__=='__main__':main()
