"""Server entry point: isolated environment, reusable data and resumable stages."""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_SHA256 = 'b3ea67fea905db5cedaec721f8d701e1f528d978fa8091568b30e2fda3f5038e'


def run(command):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using {path}. Check the running job.') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_cache(directory):
    from .data import PrepareConfig, SCHEMA_VERSION, FP_BITS, NODE_DIM, EDGE_DIM
    directory = Path(directory)
    required = ['meta.json', 'index.jsonl', 'dataset.sqlite']
    if any(not (directory / f).is_file() for f in required):
        raise ValueError(f'Incomplete cache: {directory}. Keep it for inspection and select a new --data directory.')
    meta = json.loads((directory / 'meta.json').read_text())
    if (meta['schema_version'], meta['fp_bits'], meta['node_dim'], meta['edge_dim']) != (SCHEMA_VERSION, FP_BITS, NODE_DIM, EDGE_DIM):
        raise ValueError('Prepared feature schema differs; use a new --data directory.')
    if meta['config'] != asdict(PrepareConfig()):
        raise ValueError('Cache uses different split/mass/intensity settings; use a new --data directory.')
    if meta['archive_sha256'] != ARCHIVE_SHA256 or meta['queries'] != 246 or meta['raw_records'] != 36819:
        raise ValueError('Expected the complete CASMI archive; partial smoke caches are not accepted.')
    index = (directory / 'index.jsonl').read_bytes()
    if hashlib.sha256(index).hexdigest() != meta['index_sha256']:
        raise ValueError('Prepared index checksum mismatch.')
    entries = [json.loads(line) for line in index.splitlines()]
    counts = dict(Counter(r['source'] + '/' + r['split'] for r in entries))
    if len(entries) != meta['deduplicated_records'] or counts != meta['counts']:
        raise ValueError('Prepared index counts differ from metadata.')
    if any(counts.get(f'{source}/{split}', 0) == 0 for source in ['predicted', 'experimental'] for split in ['train', 'val', 'test']):
        raise ValueError('A required source/split is empty.')
    with sqlite3.connect((directory / 'dataset.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT id FROM samples ORDER BY id').fetchall()
    if [r[0] for r in rows] != sorted(r['id'] for r in entries):
        raise ValueError('Prepared database IDs differ from the index.')
    return meta


def ensure_prepared(archive, directory, workers):
    from .data import digest
    archive, directory = Path(archive), Path(directory)
    with exclusive_lock(directory.parent / ('.' + directory.name + '.prepare.lock')):
        if directory.exists() and any(directory.iterdir()):
            meta = validate_cache(directory)
            print(f'Reusing complete cache: {directory} ({meta["deduplicated_records"]} samples)', flush=True)
            return meta
        if not archive.is_file():
            raise FileNotFoundError(f'Place casmi_top50_visualizer.zip at {archive}, or pass --archive.')
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError('Visualizer ZIP checksum differs from the supplied CASMI archive.')
        temporary = Path(tempfile.mkdtemp(prefix=directory.name + '.building-', dir=directory.parent))
        print(f'Preparing into {temporary}; publishing cache only after validation.', flush=True)
        run([sys.executable, '-m', 'casmi_neural', 'prepare', '--archive', archive,
             '--output', temporary, '--workers', str(workers)])
        meta = validate_cache(temporary)
        if directory.exists():
            directory.rmdir()  # Only the empty destination is removed.
        temporary.rename(directory)
        return meta


def check_environment(gpus=8, cpu=False):
    import numpy as np
    import rdkit
    import torch
    torch.from_numpy(np.zeros(1, dtype=np.float32))
    print(json.dumps(dict(python=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                          numpy=np.__version__, rdkit=rdkit.__version__, visible_gpus=torch.cuda.device_count()), indent=2), flush=True)
    if cpu:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        raise ValueError(f'Requested {gpus} GPUs; CUDA exposes {torch.cuda.device_count()}. Check CUDA_VISIBLE_DEVICES or --gpus.')
    for i in range(gpus):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f'GPU {i} does not support BF16.')
            x = torch.ones(8, 8, device=f'cuda:{i}', dtype=torch.bfloat16, requires_grad=True)
            (x @ x).float().sum().backward()
            torch.cuda.synchronize()
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}; BF16 forward/backward passed', flush=True)


def checkpoint_state(output, config, meta, data):
    """Validate before treating a previous stage as complete or resumable."""
    import torch
    from .model import ModelConfig, normalized_model_config
    from .train import dataset_signature
    output, data = Path(output), Path(data)
    last, best = output / 'last.pt', output / 'best.pt'
    if not last.exists():
        if best.exists():
            raise ValueError(f'{best} exists without last.pt; select another --run-root or restore the matching last.pt.')
        return 'new'
    checkpoint = torch.load(last, map_location='cpu', weights_only=True)
    if checkpoint['dataset_signature'] != dataset_signature(meta):
        raise ValueError('Checkpoint dataset signature differs; use a new --run-root.')
    if normalized_model_config(checkpoint['model_config']) != asdict(ModelConfig(**config['model'])):
        raise ValueError('Checkpoint architecture differs; use a new --run-root.')
    old_tc, new_tc = checkpoint['training_config'].copy(), config['training'].copy()
    for key in ['workers', 'log_every', 'cpu_threads', 'eval_batch_size_per_gpu', 'diagnostic_steps']:
        old_tc.pop(key, None); new_tc.pop(key, None)
    if old_tc != new_tc or checkpoint['config']['sources'] != config['sources']:
        raise ValueError('Checkpoint training settings differ; use the original configuration or a new --run-root.')
    # Also distinguish full-data training from an old --limit run.
    index = [json.loads(line) for line in (data / 'index.jsonl').read_bytes().splitlines()]
    selected = [[r for r in index if r['source'] in config['sources'] and r['split'] == split] for split in ['train', 'val']]
    signature = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    if checkpoint['subset_signature'] != signature:
        raise ValueError('Checkpoint was trained on a different source/subset; select a new --run-root.')
    if not best.is_file():
        raise ValueError(f'Missing {best}; restore the matching checkpoint before continuing.')
    complete = checkpoint['epoch'] + 1 >= new_tc['epochs'] or checkpoint['bad_epochs'] >= new_tc.get('patience', 20)
    return 'complete' if complete else 'resume'


def train_stage(stage, args, meta):
    config_path = getattr(args, stage + '_config')
    config = json.loads(config_path.read_text())
    output = args.run_root / ('neural_' + stage)
    state = checkpoint_state(output, config, meta, args.data)
    if state == 'complete':
        print(f'{stage}: already complete; using {output / "best.pt"}', flush=True)
        return
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
               f'--nproc-per-node={args.gpus}', '--master-addr', os.environ.get('MASTER_ADDR', '127.0.0.1'),
               '--master-port', str(args.port), '-m', 'casmi_neural', 'train',
               '--config', config_path, '--data', args.data, '--output', output]
    if state == 'resume':
        command += ['--resume', output / 'last.pt']
    elif stage == 'finetune':
        best = args.run_root / 'neural_pretrain' / 'best.pt'
        if not best.is_file():
            raise FileNotFoundError(f'Missing {best}; run ./aime pretrain or ./aime train first.')
        command += ['--initialize', best]
    run(command)


def parser():
    p = argparse.ArgumentParser(description='AIMe server launcher. First run: bash setup.sh; ./aime train')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['check', 'prepare', 'train', 'pretrain', 'finetune', 'evaluate', 'infer', 'status']:
        q = sub.add_parser(name)
        q.add_argument('--data', type=Path, default=ROOT / 'data/prepared/neural_v1')
        q.add_argument('--run-root', type=Path)
        q.add_argument('--profile', choices=['default','stable'], default='default',
                       help='stable: FP32, no attention checkpoint, plain AdamW, separate runs/v2_stable')
        if name == 'check':
            q.add_argument('--cpu', action='store_true', help='Only verify imports and NumPy bridge, for local development')
        if name in ['check', 'train', 'pretrain', 'finetune']:
            q.add_argument('--gpus', type=int, default=int(os.environ.get('NPROC_PER_NODE', '8')))
        if name in ['prepare', 'train', 'pretrain', 'finetune']:
            q.add_argument('--archive', type=Path, default=ROOT / 'data/raw/casmi_top50_visualizer.zip')
            q.add_argument('--workers', type=int, default=8, help='Preprocessing workers; loader settings are in the stage configs')
        if name in ['train', 'pretrain', 'finetune', 'status']:
            q.add_argument('--pretrain-config', type=Path)
            q.add_argument('--finetune-config', type=Path)
        if name in ['train', 'pretrain', 'finetune']:
            q.add_argument('--port', type=int, default=int(os.environ.get('MASTER_PORT', '29501')))
        if name in ['evaluate', 'infer']:
            q.add_argument('--checkpoint', type=Path)
            q.add_argument('--output', type=Path)
        if name == 'evaluate':
            q.add_argument('--workers', type=int, default=4)
            q.add_argument('--split', choices=['val', 'test'], default='test')
        if name == 'infer':
            q.add_argument('--sample-id', type=int, required=True)
    return p


def resolve_profile(args):
    stable = args.profile == 'stable'
    if args.run_root is None:
        args.run_root = ROOT / ('runs/v2_stable' if stable else 'runs/v2')
    for stage in ['pretrain','finetune']:
        key=stage+'_config'
        if hasattr(args,key) and getattr(args,key) is None:
            suffix='_stable' if stable else ''
            setattr(args,key,ROOT/f'configs/neural/{stage}_v2{suffix}_h20_8gpu.json')
    return args


def main():
    p = parser(); args = resolve_profile(p.parse_args())
    if getattr(args, 'gpus', 1) < 1 or getattr(args, 'workers', 1) < 1:
        p.error('--gpus and --workers must be positive')
    if not 1 <= getattr(args, 'port', 29501) <= 65535:
        p.error('--port must be between 1 and 65535')
    if args.command == 'check':
        check_environment(args.gpus, args.cpu)
    elif args.command == 'prepare':
        ensure_prepared(args.archive, args.data, args.workers)
    elif args.command in ['train', 'pretrain', 'finetune']:
        with exclusive_lock(args.run_root / '.aime-training.lock'):
            check_environment(args.gpus)
            meta = ensure_prepared(args.archive, args.data, args.workers)
            stages = ['pretrain', 'finetune'] if args.command == 'train' else [args.command]
            for stage in stages:
                train_stage(stage, args, meta)
    elif args.command == 'status':
        if not args.data.is_dir():
            print('Data: not prepared. Next: ./aime train'); return
        meta = validate_cache(args.data)
        print(f'Data: {meta["deduplicated_records"]} prepared records')
        for stage in ['pretrain', 'finetune']:
            config = json.loads(getattr(args, stage + '_config').read_text())
            state = checkpoint_state(args.run_root / ('neural_' + stage), config, meta, args.data)
            print(f'{stage}: {state}')
    else:
        check_environment(1)
        validate_cache(args.data)
        checkpoint = args.checkpoint or args.run_root / 'neural_finetune/best.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        command = [sys.executable, '-m', 'casmi_neural', args.command, '--data', args.data,
                   '--checkpoint', checkpoint, '--device', 'cuda']
        if args.command == 'evaluate':
            output = args.output or args.run_root / ('neural_finetune/' + args.split + '.json')
            command += ['--split', args.split, '--workers', str(args.workers)]
        else:
            output = args.output or args.run_root / f'peak_subnetworks_{args.sample_id}.json'
            command += ['--sample-id', str(args.sample_id)]
        output.parent.mkdir(parents=True, exist_ok=True)
        run(command + ['--output', output])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'AIMe: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
