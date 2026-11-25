from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import re
import sqlite3
import zipfile
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import SCHEMA_VERSION

RDLogger.DisableLog('rdApp.warning')
ELEMENTS = ('C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se')
FP_BITS = 2048
NODE_DIM = len(ELEMENTS) + 9
EDGE_DIM = len(ELEMENTS) + 2
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS, includeChirality=True)


@dataclass(frozen=True)
class PrepareConfig:
    tolerance_da: float = .007
    tolerance_ppm: float = 5.
    intensity_power: float = .5
    split_seed: int = 20261016
    split: str = 'scaffold'


def tolerance(mass, cfg):
    return max(cfg.tolerance_da, mass * cfg.tolerance_ppm * 1e-6)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@lru_cache(maxsize=100000)
def molecule(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    bits = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(_GEN.GetFingerprint(m), bits)
    canonical = Chem.MolToSmiles(m, isomericSmiles=False)
    family = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or 'acyclic:' + canonical
    return np.packbits(bits, bitorder='little'), m.GetNumHeavyAtoms(), Chem.GetFormalCharge(m), canonical, family


def formula_vector(formula):
    counts = Counter()
    for element, number in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        counts[element] += int(number or 1)
    unknown = set(counts) - set(ELEMENTS)
    if unknown:
        raise ValueError(f'Unsupported formula elements {unknown}; extend ELEMENTS and rebuild cache')
    return np.array([counts[e] for e in ELEMENTS], dtype=np.float32)


def entry_record(entry, source, cfg):
    """Only structures, graph topology, masses, spectrum and conditions are inputs.

    No fragment_matching, peak_matchings, final_prob, action probabilities,
    target flags, known reference molecule or similarity scores are read here.
    """
    nodes = entry['dag_nodes']
    if not nodes:
        raise ValueError('Empty DAG')
    lookup = {n['nid']: i for i, n in enumerate(nodes)}
    if len(lookup) != len(nodes):
        raise ValueError('Duplicate node IDs')
    edges = []
    indegree = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        if len(set(n['out_nids'])) != len(n['out_nids']):
            raise ValueError('Repeated DAG edge')
        for child in n['out_nids']:
            if child not in lookup or n['nid'] not in nodes[lookup[child]]['in_nids']:
                raise ValueError('Dangling/one-sided DAG edge')
            j = lookup[child]; edges.append((i, j)); indegree[j] += 1
        for parent in n['in_nids']:
            if parent not in lookup or n['nid'] not in nodes[lookup[parent]]['out_nids']:
                raise ValueError('Dangling/one-sided DAG parent')
    roots = np.flatnonzero(indegree == 0)
    if len(roots) != 1:
        raise ValueError(f'Expected one DAG root, got {len(roots)}')
    level = np.zeros(len(nodes), dtype=np.int64)
    todo = deque(roots.tolist()); remain = indegree.copy(); seen = 0
    children = [[] for _ in nodes]
    for i, j in edges:
        children[i].append(j)
    while todo:
        i = todo.popleft(); seen += 1
        for j in children[i]:
            level[j] = max(level[j], level[i] + 1)
            remain[j] -= 1
            if remain[j] == 0:
                todo.append(j)
    if seen != len(nodes):
        raise ValueError('Graph is cyclic')
    masses = np.array([n['mass'] for n in nodes], dtype=np.float64)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('Invalid node masses')
    precursor = float(entry['experiment_metadata']['precursor_mass'])
    formulas = np.stack([formula_vector(n['formula']) for n in nodes])
    fps, numeric = [], []
    for i, n in enumerate(nodes):
        fp, heavy, charge, _, _ = molecule(n['SMILES'])
        fps.append(fp)
        numeric.append(np.concatenate([formulas[i] / 50., [
            masses[i] / precursor, float(n.get('total_num_h', 0)) / 100., heavy / 100., charge / 3.,
            np.log1p(level[i]) / 4., np.log1p(indegree[i]) / 4.,
            np.log1p(len(children[i])) / 4., float(i == roots[0]), float(not children[i])]]))
    edge_index = np.array(edges, dtype=np.int64).reshape(-1, 2).T
    edge_attr = np.array([np.concatenate([(formulas[i] - formulas[j]) / 50.,
                           [(masses[i] - masses[j]) / precursor, masses[j] / masses[i]]])
                          for i, j in edges], dtype=np.float32).reshape(-1, EDGE_DIM)
    # The mass readout is fixed by the DAG. Merge only against a fixed cluster
    # anchor; no chaining and no observed-intensity-dependent mass bins.
    bin_masses, node_bin = [], np.zeros(len(nodes), dtype=np.int64)
    for i in np.argsort(masses, kind='stable'):
        if not bin_masses or masses[i] - bin_masses[-1] > tolerance(bin_masses[-1], cfg):
            bin_masses.append(float(masses[i]))
        node_bin[i] = len(bin_masses) - 1
    peaks = np.array(entry[f'{source}_ms2'], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(peaks).all() or np.any(peaks[:, 0] <= 0) or np.any(peaks[:, 1] < 0):
        raise ValueError('Invalid spectrum')
    peaks = peaks[peaks[:, 1] > 0]
    if len(peaks) == 0:
        raise ValueError('Empty positive-intensity spectrum')
    peaks = peaks[np.argsort(peaks[:, 0], kind='stable')]
    peak_bin = np.full(len(peaks), len(bin_masses), dtype=np.int64)
    grid = np.array(bin_masses)
    # Compare each observed peak to all DAG masses, not teacher peak mappings.
    for i, (mz, _) in enumerate(peaks):
        compatible = np.flatnonzero(np.abs(grid - mz) <= tolerance(mz, cfg))
        if len(compatible):
            peak_bin[i] = compatible[np.argmin(np.abs(grid[compatible] - mz))]
    signal = peaks[:, 1] ** cfg.intensity_power
    target = np.bincount(peak_bin, weights=signal, minlength=len(grid) + 1)
    target /= target.sum()
    md = entry['experiment_metadata']
    ion = md['precursor_type']
    if ion not in ('[M+H]+', '[M-H]-'):
        raise ValueError(f'Unsupported adduct {ion}; extend ion features before training')
    packed = {
        'node_fp': torch.from_numpy(np.stack(fps)),
        # Optional provenance for diagnostics, not an encoder/readout feature.
        # Old caches remain usable with conservative ratio-rounding bounds.
        'node_masses': torch.from_numpy(masses.copy()),
        'node_num': torch.tensor(np.array(numeric), dtype=torch.float32),
        'edge_index': torch.from_numpy(edge_index), 'edge_attr': torch.from_numpy(edge_attr),
        'levels': torch.from_numpy(level), 'root': int(roots[0]),
        'node_bin': torch.from_numpy(node_bin), 'bin_masses': torch.from_numpy(grid),
        'peaks': torch.from_numpy(peaks), 'peak_bin': torch.from_numpy(peak_bin),
        'target': torch.tensor(target, dtype=torch.float32),
        'meta': torch.tensor([float(md['collision_energy_nce']) / 100.,
                              1. if ion == '[M+H]+' else -1., precursor / 1000.]),
        'nids': [int(n['nid']) for n in nodes], 'smiles': [n['SMILES'] for n in nodes],
        'formulas': [n['formula'] for n in nodes], 'precursor_mz': precursor}
    return packed


def split_name(group, seed):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:12], 16) % 100
    return 'train' if value < 80 else 'val' if value < 90 else 'test'


def _prepare_query(args):
    archive, item, cfg = args
    with zipfile.ZipFile(archive) as z:
        raw = z.read(item['file'])
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError(f'Checksum failed: {item["file"]}')
    entries = json.loads(raw)
    if len(entries) != item['entries']:
        raise ValueError('Entry count mismatch')
    output = []
    for e in entries:
        _, _, _, identity, family = molecule(e['pred_mol_metadata']['smiles'])
        group = family if cfg.split == 'scaffold' else identity
        for source in ['predicted', 'experimental']:
            # This selects a known precursor/spectrum pair, NOT peak annotations.
            if source == 'experimental' and not e['research_metadata']['is_target']:
                continue
            record = entry_record(e, source, cfg)
            info = {'qid': str(e['identifiers']['query_id']), 'spectrum_id': str(e['identifiers']['id']),
                    'candidate_index': int(e['research_metadata']['candidate_index']),
                    'source': source, 'group': group, 'split': split_name(group, cfg.split_seed),
                    'num_nodes': len(e['dag_nodes']), 'num_edges': record['edge_index'].shape[1],
                    'num_peaks': len(record['peaks']), 'other_target': float(record['target'][-1])}
            if source == 'predicted':
                key = json.dumps([source, identity, e['experiment_metadata']['collision_energy_nce'],
                                  e['experiment_metadata']['precursor_type'], e['experiment_metadata'].get('instrument')])
            else:
                key = source + ':' + info['spectrum_id']
            record['info'] = info
            stream = io.BytesIO(); torch.save(record, stream)
            output.append((key, info['num_nodes'], zlib.compress(stream.getvalue(), level=1), json.dumps(info)))
    return output


def prepare(archive, output, cfg, workers=1, max_queries=None):
    archive, output = Path(archive), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} must be empty; use a new directory')
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        items = sorted(json.loads(z.read('export_report.json'))['queries'], key=lambda r:int(r['qid']))
    if max_queries:
        items = items[:max_queries]
    db = sqlite3.connect(output / 'dataset.sqlite')
    db.execute('CREATE TABLE samples (id INTEGER PRIMARY KEY, key TEXT UNIQUE, quality INTEGER, payload BLOB, info TEXT)')
    sql = ('INSERT INTO samples(key,quality,payload,info) VALUES (?,?,?,?) ON CONFLICT(key) DO UPDATE SET '
           'quality=excluded.quality,payload=excluded.payload,info=excluded.info WHERE excluded.quality>samples.quality')
    jobs = [(str(archive), item, cfg) for item in items]
    if workers > 1:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'))
        pending = deque(); job_iter = iter(jobs)
        for _ in range(min(len(jobs), workers * 2)):
            pending.append(pool.submit(_prepare_query, next(job_iter)))
        def results():
            while pending:
                value = pending.popleft().result()
                job = next(job_iter, None)
                if job is not None:
                    pending.append(pool.submit(_prepare_query, job))
                yield value
        iterator = results()
    else:
        pool = None; iterator = map(_prepare_query, jobs)
    raw_count = 0
    try:
        for i, rows in enumerate(iterator):
            db.executemany(sql, rows); db.commit(); raw_count += len(rows)
            if i % 10 == 0 or i + 1 == len(jobs):
                print(f'Prepared {i+1}/{len(jobs)} queries', flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    index = []
    for sid, text in db.execute('SELECT id,info FROM samples ORDER BY id'):
        index.append({'id': sid, **json.loads(text)})
    db.close()
    index_bytes = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in index).encode()
    (output / 'index.jsonl').write_bytes(index_bytes)
    meta = {'schema_version': SCHEMA_VERSION, 'config': asdict(cfg), 'fp_bits': FP_BITS,
            'node_dim': NODE_DIM, 'edge_dim': EDGE_DIM, 'queries': len(items),
            'raw_records': raw_count, 'deduplicated_records': len(index),
            'archive_sha256': digest(archive), 'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
            'counts': dict(Counter(r['source'] + '/' + r['split'] for r in index)),
            'group_counts': {s:len({r['group'] for r in index if r['split']==s}) for s in ['train','val','test']},
            'prepare_code_sha256': digest(__file__)}
    (output / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)


class SpectrumDAGDataset(Dataset):
    def __init__(self, directory, sources=('predicted',), split='train', limit=None):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'meta.json').read_text())
        if self.meta['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Cache schema mismatch; rebuild prepared data')
        if (self.meta['fp_bits'],self.meta['node_dim'],self.meta['edge_dim']) != (FP_BITS,NODE_DIM,EDGE_DIM):
            raise ValueError('Chemical feature dimensions changed; rebuild prepared data')
        index_bytes = (self.directory / 'index.jsonl').read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != self.meta['index_sha256']:
            raise ValueError('Index checksum mismatch')
        self.index = [r for line in index_bytes.splitlines() if (r := json.loads(line))['source'] in sources
                      and (split == 'all' or r['split'] == split)]
        if limit:
            self.index = self.index[:limit]
        self._db = None

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy(); state['_db'] = None; return state

    def __getitem__(self, index):
        if self._db is None:
            self._db = sqlite3.connect(f'file:{self.directory / "dataset.sqlite"}?mode=ro', uri=True)
        sid = self.index[index]['id']
        row = self._db.execute('SELECT payload FROM samples WHERE id=?', (sid,)).fetchone()
        data = torch.load(io.BytesIO(zlib.decompress(row[0])), map_location='cpu', weights_only=True)
        data['id'] = sid
        return data


def collate(records):
    batch = {'records': records, 'ids': [r.get('id',i) for i,r in enumerate(records)]}
    lists = {k:[] for k in ['node_fp','node_num','edge_index','edge_attr','levels','node_bin','target',
                           'node_batch','bin_batch','roots','others','meta','bin_masses']}
    n_offset = b_offset = 0
    max_peaks = max(len(r['peaks']) for r in records)
    tokens = torch.zeros(len(records), max_peaks, 2)
    valid = torch.zeros(len(records), max_peaks, dtype=torch.bool)
    token_bins = torch.full((len(records), max_peaks), -1, dtype=torch.long)
    for b, r in enumerate(records):
        n, nb = len(r['node_num']), len(r['target'])
        fp = np.unpackbits(r['node_fp'].numpy(), axis=1, bitorder='little').astype(np.float32)
        lists['node_fp'].append(torch.from_numpy(fp)); lists['node_num'].append(r['node_num'])
        lists['edge_index'].append(r['edge_index'] + n_offset); lists['edge_attr'].append(r['edge_attr'])
        lists['levels'].append(r['levels']); lists['node_bin'].append(r['node_bin'] + b_offset)
        lists['target'].append(r['target']); lists['node_batch'].append(torch.full((n,), b, dtype=torch.long))
        lists['bin_batch'].append(torch.full((nb,), b, dtype=torch.long))
        lists['bin_masses'].append(torch.cat([r['bin_masses'],torch.tensor([float('nan')])]))
        lists['roots'].append(r['root'] + n_offset); lists['others'].append(b_offset + nb - 1)
        lists['meta'].append(r['meta'])
        p = len(r['peaks']); tokens[b,:p,0] = r['peaks'][:,0] / r['precursor_mz']
        tokens[b,:p,1] = r['peaks'][:,1].float(); valid[b,:p] = True
        token_bins[b,:p] = r['peak_bin'] + b_offset
        n_offset += n; b_offset += nb
    for k, values in lists.items():
        batch[k] = torch.stack(values) if k=='meta' else torch.tensor(values) if k in ['roots','others'] else torch.cat(values,dim=1 if k=='edge_index' else 0)
    batch.update(tokens=tokens, token_valid=valid, token_bins=token_bins)
    # Source-level masks are precomputed on CPU: no per-level GPU sync during flow.
    source_levels = batch['levels'][batch['edge_index'][0]]
    batch['edge_layers'] = [torch.nonzero(source_levels == level).flatten()
                            for level in range(int(batch['levels'].max())+1)]
    batch['num_graphs'] = len(records)
    counts=torch.tensor([len(r['node_num']) for r in records],dtype=torch.long)
    batch['node_counts']=counts
    batch['node_local']=torch.cat([torch.arange(int(n)) for n in counts])
    batch['max_nodes']=int(counts.max())
    return batch


def visibility(batch, ratio, seed):
    visible = batch['token_valid'].clone()
    if ratio <= 0:
        batch['visible'] = visible; return batch
    for b, sid in enumerate(batch['ids']):
        bins = batch['token_bins'][b, batch['token_valid'][b]].unique()
        g = torch.Generator().manual_seed((seed + int(sid) * 1000003) % (2**63-1))
        count = max(1, round(len(bins) * ratio))
        if len(bins) > 1:
            count = min(count, len(bins)-1)
        hidden = bins[torch.randperm(len(bins), generator=g)[:count]]
        visible[b] &= ~torch.isin(batch['token_bins'][b], hidden)
    batch['visible'] = visible
    return batch


def to_device(batch, device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor)
            else [x.to(device,non_blocking=True) for x in v] if k=='edge_layers' else v
            for k,v in batch.items()}
