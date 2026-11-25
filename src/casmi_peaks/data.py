"""Teacher-label extraction, train-only vocabulary, and spectrum-only student inputs."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass,asdict
from functools import partial
import hashlib
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sqlite3
import tempfile
import zipfile
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset
from . import SCHEMA_VERSION
from .chem import structure,graph_features,ion_sign,ATOM_DIM,BOND_DIM,NUMERIC_DIM


@dataclass(frozen=True)
class PrepareConfig:
    tolerance_da:float=.007
    tolerance_ppm:float=5.
    exclude_precursor:bool=True


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def tolerance(mz,cfg):return max(cfg.tolerance_da,mz*cfg.tolerance_ppm*1e-6)


def _query(job):
    archive,qid,reference,cfg=job
    with zipfile.ZipFile(archive) as z:entries=json.loads(z.read(f'visualizer/casmi_{qid}.json'))
    lookup={}
    for entry in entries:
        key=(str(entry['identifiers']['id']),int(entry['research_metadata']['candidate_index']))
        if key in lookup:raise ValueError(f'Duplicate archive identity: {key}')
        lookup[key]=entry
    records=[];labels={};audit=Counter()
    for info in reference:
        entry=lookup[(info['spectrum_id'],info['candidate_index'])]
        if info['source']=='experimental' and not entry['research_metadata']['is_target']:
            raise ValueError('Experimental label DAG must be the target precursor')
        md=entry['experiment_metadata'];ion=md['precursor_type'];ion_sign(ion)
        precursor=float(md['precursor_mass']);energy=float(md['collision_energy_nce'])
        raw_peaks=entry[info['source']+'_ms2']
        if not math.isfinite(precursor) or precursor<=0 or not math.isfinite(energy):raise ValueError('Invalid conditions')
        indexed=[]
        for i,(mz,intensity) in enumerate(raw_peaks):
            if not all(math.isfinite(float(v)) for v in [mz,intensity]) or mz<=0 or intensity<0:raise ValueError('Invalid peak')
            if intensity>0:indexed.append((i,float(mz),float(intensity)))
        indexed.sort(key=lambda v:v[1])
        if not indexed:raise ValueError('Empty spectrum')
        reverse={old:new for new,(old,_,_) in enumerate(indexed)}
        peaks=[[mz,intensity] for _,mz,intensity in indexed]
        targets=[defaultdict(float) for _ in peaks]
        precursor_peak=[abs(mz-precursor)<=tolerance(mz,cfg) for mz,_ in peaks]
        mapping=entry.get('fragment_matching')
        if mapping is None or len(mapping)!=len(entry['dag_nodes']):raise ValueError('Missing node-to-predicted-peak mapping')
        for node,mapped in zip(entry['dag_nodes'],mapping):
            audit['nodes']+=1
            if not node['in_nids']:continue  # Root is not a fragment label.
            mass=float(node['mass'])
            if not math.isfinite(mass) or mass<=0:raise ValueError('Invalid fragment mass')
            try:label=structure(node['SMILES'],node['formula'],ion)
            except ValueError as exc:
                audit['excluded_invalid_or_heavy_atom_inconsistent']+=1;continue
            audit['formula_smiles_disagreement']+=not label['formula_smiles_consistent']
            key=label['key']
            if info['split']=='train':
                if key not in labels:labels[key]={**label,'mz_min':mass,'mz_max':mass,'mz_sum':0.,'mass_count':0,'train_peak_occurrences':0}
                item=labels[key];item['mz_min']=min(item['mz_min'],mass);item['mz_max']=max(item['mz_max'],mass)
                item['mz_sum']+=mass;item['mass_count']+=1
            if info['source']=='predicted':
                mapped=int(mapped)
                if not 0<=mapped<len(entry['predicted_ms2']):raise ValueError('Invalid fragment_matching index')
                mz=float(entry['predicted_ms2'][mapped][0])
                if abs(mz-mass)>tolerance(mz,cfg):raise ValueError('Teacher mapping violates mass tolerance')
                indices=[reverse[mapped]] if mapped in reverse else []
            else:
                indices=[i for i,(mz,_) in enumerate(peaks) if abs(mz-mass)<=tolerance(mz,cfg)]
            weight=float(node.get('final_prob',0.))
            if not math.isfinite(weight) or weight<0:raise ValueError('Invalid teacher contribution')
            for i in indices:
                if cfg.exclude_precursor and precursor_peak[i]:continue
                targets[i][key]+=weight if info['source']=='predicted' else 1.
        packed_targets=[]
        for target in targets:
            if target:
                if info['split']=='train':
                    for key in target:labels[key]['train_peak_occurrences']+=1
                total=sum(target.values())
                if total<=0:audit['uniform_zero_teacher_weight_peaks']+=1
                packed_targets.append([[k,v/total if total>0 else 1/len(target)] for k,v in sorted(target.items())])
            else:packed_targets.append([])
        record=dict(id=info['id'],source=info['source'],split=info['split'],qid=info['qid'],
            spectrum_id=info['spectrum_id'],candidate_index=info['candidate_index'],group=info['group'],
            precursor_mz=precursor,collision_energy_nce=energy,precursor_type=ion,
            peaks=peaks,targets=packed_targets,precursor_peak=precursor_peak)
        audit['spectra']+=1;audit['peaks']+=len(peaks)
        audit['teacher_annotated_peaks']+=sum(bool(t) for t in targets)
        audit['precursor_peaks']+=sum(precursor_peak)
        payload=zlib.compress(json.dumps(record,separators=(',',':'),allow_nan=False).encode())
        records.append((info['id'],info['source'],info['split'],info['qid'],payload,len(peaks),sum(bool(t) for t in targets)))
    return records,labels,dict(audit)


def prepare(archive,reference,output,workers=4,max_queries=None):
    """Reference index determines exact legacy dedup/splits, not student features."""
    from casmi_neural.train import dataset_signature
    reference,archive,output=Path(reference),Path(archive),Path(output)
    meta=json.loads((reference/'meta.json').read_text());index_bytes=(reference/'index.jsonl').read_bytes()
    if digest(archive)!=meta['archive_sha256']:raise ValueError('Archive differs from reference index')
    if hashlib.sha256(index_bytes).hexdigest()!=meta['index_sha256']:raise ValueError('Reference index checksum mismatch')
    if meta['config']['tolerance_da']!=.007 or meta['config']['tolerance_ppm']!=5.:raise ValueError('Unexpected reference tolerances')
    records=[json.loads(line) for line in index_bytes.splitlines()]
    if len(records)!=meta['deduplicated_records']:raise ValueError('Reference record count mismatch')
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate sample IDs')
    strict_protocol=meta['config'].get('split')=='query_scaffold_intersection_v1'
    if strict_protocol:
        from casmi_joint.protocol import validate_query_reference
        validate_query_reference(reference)
        query_splits=defaultdict(set)
        for row in records: query_splits[row['qid']].add(row['split'])
        if any(len(s)!=1 for s in query_splits.values()): raise ValueError('Query leakage in strict reference')
    by_query=defaultdict(list);groups=defaultdict(set)
    for r in records:by_query[r['qid']].append(r);groups[r['group']].add(r['split'])
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Scaffold leakage in reference index')
    qids=sorted(by_query,key=int)
    if max_queries is not None:qids=qids[:max_queries]
    if output.exists() and any(output.iterdir()):raise ValueError('Output is not empty; select another --data directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    cfg=PrepareConfig()
    temporary=Path(tempfile.mkdtemp(prefix=output.name+'.building-',dir=output.parent))
    labels={};audit=Counter();counts=Counter();index=[]
    jobs=[(str(archive),qid,by_query[qid],cfg) for qid in qids]
    pool=ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) if workers>1 else None
    try:
        results=pool.map(_query,jobs) if pool else map(_query,jobs)
        with sqlite3.connect(temporary/'records.sqlite') as db:
            db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY,source TEXT,split TEXT,qid TEXT,payload BLOB)')
            for done,(rows,local,stats) in enumerate(results,1):
                db.executemany('INSERT INTO records VALUES (?,?,?,?,?)',[r[:5] for r in rows]);db.commit()
                for sid,source,split,qid,_,npeaks,nlabels in rows:
                    counts[source+'/'+split]+=1
                    index.append(dict(id=sid,source=source,split=split,qid=qid,num_peaks=npeaks,annotated_peaks=nlabels))
                for key,v in local.items():
                    if key not in labels:labels[key]=v
                    else:
                        old=labels[key];old['mz_min']=min(old['mz_min'],v['mz_min']);old['mz_max']=max(old['mz_max'],v['mz_max'])
                        for field in ['mz_sum','mass_count','train_peak_occurrences']:old[field]+=v[field]
                audit.update(stats)
                if done%10==0 or done==len(jobs):print(json.dumps(dict(prepared_queries=done,total=len(jobs),vocabulary=len(labels))),flush=True)
        db.close()
        vocab=[]
        for key,v in sorted(labels.items()):
            v=dict(v);v['mz']=v.pop('mz_sum')/v['mass_count'];v['id']=len(vocab);vocab.append(v)
        if not vocab:raise ValueError('No training vocabulary')
        (temporary/'vocab.json').write_text(json.dumps(vocab,separators=(',',':'))+'\n')
        atoms=[];edges=[];bonds=[];numeric=[];node_ptr=[0];edge_ptr=[0]
        for label in vocab:
            a,e,b,n=graph_features(label);atoms.append(a);edges.append(e);bonds.append(b);numeric.append(n)
            node_ptr.append(node_ptr[-1]+len(a));edge_ptr.append(edge_ptr[-1]+e.shape[1])
        torch.save(dict(atoms=torch.cat(atoms),edges=torch.cat(edges,1),bonds=torch.cat(bonds),numeric=torch.stack(numeric),
            node_ptr=torch.tensor(node_ptr),edge_ptr=torch.tensor(edge_ptr)),temporary/'features.pt')
        index.sort(key=lambda r:r['id'])
        (temporary/'index.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in index))
        result=dict(schema_version=SCHEMA_VERSION,task='DAG-free spectrum-to-fragment retrieval',
            label_semantics='nonstereochemical AIMe-exported graph + reported ion formula + precursor ion mode; not confirmed ion structures',
            config=asdict(cfg),reference_signature=dataset_signature(meta),
            partial=max_queries is not None or meta.get('queries')!=246 or
                (not (meta.get('protocol_complete') and meta.get('source_complete')) if strict_protocol else len(records)!=36279),
            queries=len(qids),counts=dict(counts),audit=dict(audit),vocabulary=len(vocab),
            atom_dim=ATOM_DIM,bond_dim=BOND_DIM,numeric_dim=NUMERIC_DIM,
            hashes={name:digest(temporary/name) for name in ['vocab.json','features.pt','index.jsonl','records.sqlite']})
        (temporary/'meta.json').write_text(json.dumps(result,indent=2)+'\n')
        if output.exists():output.rmdir()
        temporary.rename(output)
        return result
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)


def signature(meta):
    return {k:meta[k] for k in ['schema_version','label_semantics','config','reference_signature','partial','hashes']}


class PeakDataset(Dataset):
    def __init__(self,directory,sources,split,limit=None):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'meta.json').read_text())
        if self.meta['schema_version']!=SCHEMA_VERSION:raise ValueError('Peak cache schema mismatch')
        for name in ['index.jsonl','records.sqlite']:
            if digest(self.directory/name)!=self.meta['hashes'][name]:raise ValueError(f'Checksum mismatch: {name}')
        self.index=[json.loads(line) for line in (self.directory/'index.jsonl').read_text().splitlines()]
        self.index=[r for r in self.index if r['source'] in sources and r['split']==split]
        if limit is not None:self.index=self.index[:limit]
        self._db=None

    def __len__(self):return len(self.index)

    def __getstate__(self):
        state=self.__dict__.copy();state['_db']=None;return state

    def __getitem__(self,i):
        if self._db is None:self._db=sqlite3.connect((self.directory/'records.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        raw=self._db.execute('SELECT payload FROM records WHERE id=?',(self.index[i]['id'],)).fetchone()
        if raw is None:raise ValueError('Missing spectrum record')
        return json.loads(zlib.decompress(raw[0]))

    def close(self):
        if self._db is not None:self._db.close();self._db=None

    def __del__(self):
        db=getattr(self,'_db',None)
        if db is not None:db.close()


def spectrum_batch(records):
    """No structures, labels, target flags or DAG features enter the spectrum encoder."""
    p=max(len(r['peaks']) for r in records)
    peaks=torch.zeros(len(records),p,2);valid=torch.zeros(len(records),p,dtype=torch.bool);conditions=[]
    for b,r in enumerate(records):
        x=torch.tensor(r['peaks'],dtype=torch.float32);peaks[b,:len(x)]=x;valid[b,:len(x)]=True
        conditions.append([r['collision_energy_nce']/100.,r['precursor_mz']/1000.,ion_sign(r['precursor_type'])])
    return dict(peaks=peaks,valid=valid,conditions=torch.tensor(conditions,dtype=torch.float32))


class Catalog:
    def __init__(self,directory,load_features=True):
        self.directory=Path(directory);meta=json.loads((self.directory/'meta.json').read_text())
        for name in ['vocab.json']+(['features.pt'] if load_features else []):
            if digest(self.directory/name)!=meta['hashes'][name]:raise ValueError(f'Catalog checksum mismatch: {name}')
        self.labels=json.loads((self.directory/'vocab.json').read_text())
        self.keys={v['key']:i for i,v in enumerate(self.labels)}
        self.features=torch.load(self.directory/'features.pt',map_location='cpu',weights_only=True,mmap=True) if load_features else None
        self.cfg=PrepareConfig(**meta['config']);self.tables={}
        self.freq=np.array([v['train_peak_occurrences'] for v in self.labels])
        self.masses=np.array([v['mz'] for v in self.labels])
        self.minimum=np.array([v['mz_min'] for v in self.labels]);self.maximum=np.array([v['mz_max'] for v in self.labels])
        self.radius=float(np.maximum(self.masses-self.minimum,self.maximum-self.masses).max())
        for ion in ['[M+H]+','[M-H]-']:
            ids=np.array([i for i,v in enumerate(self.labels) if v['ion']==ion],dtype=np.int64)
            ids=ids[np.argsort(self.masses[ids],kind='stable')];self.tables[ion]=(self.masses[ids],ids)

    def compatible(self,mz,ion):
        ion_sign(ion);mass,ids=self.tables[ion];tol=tolerance(mz,self.cfg)
        a=np.searchsorted(mass,mz-tol-self.radius,'left');b=np.searchsorted(mass,mz+tol+self.radius,'right')
        chosen=ids[a:b]
        return chosen[(self.minimum[chosen]-tol<=mz)&(self.maximum[chosen]+tol>=mz)].tolist()

    def graphs(self,ids):
        f=self.features;atoms=[];edges=[];bonds=[];groups=[];offset=0
        for b,i in enumerate(ids):
            lo,hi=map(int,f['node_ptr'][i:i+2]);a,z=map(int,f['edge_ptr'][i:i+2])
            atoms.append(f['atoms'][lo:hi]);edges.append(f['edges'][:,a:z]+offset);bonds.append(f['bonds'][a:z])
            groups.append(torch.full((hi-lo,),b,dtype=torch.long));offset+=hi-lo
        return dict(atoms=torch.cat(atoms),edges=torch.cat(edges,1),bonds=torch.cat(bonds),
                    groups=torch.cat(groups),numeric=f['numeric'][ids],size=len(ids))


def training_targets(records,catalog,rng,max_peaks=32,negatives=64):
    selected=[]
    for b,r in enumerate(records):
        eligible=[i for i,t in enumerate(r['targets']) if any(k in catalog.keys for k,_ in t)]
        if len(eligible)>max_peaks:eligible=rng.sample(eligible,max_peaks)
        for i in eligible:
            allowed=catalog.compatible(r['peaks'][i][0],r['precursor_type']);pool=set(allowed)
            positive={catalog.keys[k]:w for k,w in r['targets'][i] if k in catalog.keys and catalog.keys[k] in pool}
            if not positive:continue
            negative=[k for k in allowed if k not in positive]
            if len(negative)>negatives:negative=rng.sample(negative,negatives)
            ids=sorted(positive)+negative
            selected.append((b,i,ids,positive))
    if not selected:return None
    unique=sorted({k for _,_,ids,_ in selected for k in ids});lookup={k:i for i,k in enumerate(unique)}
    width=max(len(ids) for _,_,ids,_ in selected)
    indices=torch.zeros(len(selected),width,dtype=torch.long);valid=torch.zeros_like(indices,dtype=torch.bool)
    positive=torch.zeros_like(valid);weights=torch.zeros_like(indices,dtype=torch.float32);positions=[]
    for j,(b,i,ids,target) in enumerate(selected):
        positions.append([b,i]);indices[j,:len(ids)]=torch.tensor([lookup[k] for k in ids]);valid[j,:len(ids)]=True
        for n,k in enumerate(ids):
            if k in target:positive[j,n]=True;weights[j,n]=target[k]
        weights[j]/=weights[j].sum().clamp_min(1e-12)
    return dict(ids=unique,positions=torch.tensor(positions),indices=indices,valid=valid,positive=positive,weights=weights)
