"""Per-rank native crash traces and bounded, synchronized startup milestones."""
from contextlib import contextmanager
import faulthandler
import json
import os
from pathlib import Path
import sys
import time

import torch


class FlightRecorder:
    def __init__(self, output):
        self.rank = int(os.environ.get('RANK', '0'))
        self.directory = Path(output) / 'diagnostics'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / f'rank_{self.rank}.jsonl').open('a', buffering=1)

    def mark(self, stage, device=None, **fields):
        event = dict(time=time.time(), pid=os.getpid(), rank=self.rank, stage=stage, **fields)
        # Flush BEFORE synchronization, so a failure inside sync has a location.
        self.stream.write(json.dumps(event) + '\n')
        print(f'[rank {self.rank}] {stage}' + (f' epoch={fields["epoch"]} batch={fields["batch"]}'
              if 'epoch' in fields and 'batch' in fields else ''), flush=True)
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)
        self.stream.write(json.dumps(dict(event, stage=stage + ':complete')) + '\n')

    def batch(self, batch, epoch, step, mask_seed):
        self.mark('batch_loaded_cpu', epoch=epoch, batch=step + 1,
                  sample_ids=list(map(int, batch['ids'])), mask_seed=mask_seed,
                  graphs=batch['num_graphs'], nodes=len(batch['node_num']),
                  edges=batch['edge_index'].shape[1], max_nodes=batch['max_nodes'],
                  peak_shape=list(batch['tokens'].shape))


@contextmanager
def native_diagnostics(output):
    recorder = FlightRecorder(output)
    fault = (recorder.directory / f'rank_{recorder.rank}.fault.log').open('a', buffering=1)
    was_enabled = faulthandler.is_enabled()
    faulthandler.enable(file=fault, all_threads=True)
    try:
        recorder.mark('process_start', torch=str(torch.__version__),
                      cuda=torch.version.cuda, python=sys.version.split()[0])
        yield recorder
    except BaseException as exc:
        recorder.mark('python_exception', error=repr(exc))
        raise
    finally:
        # A fatal native signal terminates before this block; the FD stays open.
        if was_enabled:
            faulthandler.enable(all_threads=True)
        else:
            faulthandler.disable()
        fault.close()
        recorder.stream.close()


def configure_cuda(training):
    """Only select explicitly requested backend settings; preserve old defaults."""
    torch.backends.cuda.matmul.allow_tf32 = training.get('allow_tf32', True)
    if 'cudnn_benchmark' in training:
        torch.backends.cudnn.benchmark = training['cudnn_benchmark']
    if 'cudnn_deterministic' in training:
        torch.backends.cudnn.deterministic = training['cudnn_deterministic']
    if training.get('disable_fused_sdpa', False):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def optimizer_options(training):
    # None preserves the existing PyTorch choice for old configurations.
    return dict(foreach=training.get('optimizer_foreach'), fused=training.get('optimizer_fused'))
