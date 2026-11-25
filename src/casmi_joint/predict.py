"""Common inference path for evaluation and deployment; teacher fields are ignored."""
import math
import numpy as np
import torch
from casmi_peaks.data import spectrum_batch,tolerance
from casmi_peaks.engine import move,_rank
from .model import joint_update


def observed_pools(record,catalog):
    """Use every observed non-precursor peak, including unannotated peaks."""
    return [[] if abs(mz-record['precursor_mz'])<=tolerance(mz,catalog.cfg)
            else catalog.compatible(mz,record['precursor_type']) for mz,_ in record['peaks']]


class CatalogSubsetEmbeddings:
    """Catalog-ID indexed view of an explicitly encoded observation-only subset."""
    def __init__(self,ids,values,catalog_size):
        self.ids=np.asarray(ids,dtype=np.int64); self.values=np.asarray(values,dtype=np.float32)
        if len(self.ids)!=len(self.values) or len(set(self.ids.tolist()))!=len(self.ids):
            raise ValueError('Invalid subset embedding IDs')
        if (self.ids<0).any() or (self.ids>=catalog_size).any(): raise ValueError('Subset ID outside catalog')
        self.lookup=np.full(catalog_size,-1,dtype=np.int64)
        self.lookup[self.ids]=np.arange(len(self.ids))

    def __getitem__(self,ids):
        positions=self.lookup[np.asarray(ids,dtype=np.int64)]
        if (positions<0).any(): raise ValueError('Requested candidate was not encoded in observation subset')
        return self.values[positions]


@torch.no_grad()
def encode_observed_catalog(model,catalog,records,device,batch_size=256):
    """Encode candidates needed by these records; no targets/DAGs are inspected.

    Rebuild on every evaluation invocation because the fragment encoder can
    change between training epochs. Ranks maintain independent exact subsets.
    """
    ids=sorted({k for record in records for pool in observed_pools(record,catalog) for k in pool})
    chunks=[]
    for start in range(0,len(ids),batch_size):
        chunks.append(model.fragment(move(catalog.graphs(ids[start:start+batch_size]),device)).cpu())
    values=torch.cat(chunks).numpy() if chunks else np.empty((0,model.cfg.hidden),dtype=np.float32)
    return CatalogSubsetEmbeddings(ids,values,len(catalog.labels))


@torch.no_grad()
def prepare_record(model,record,catalog,device,settings,embeddings=None,query=None):
    """Compute unary retrieval once for any number of relation-strength trials."""
    if query is None: query=model.encode_spectrum(move(spectrum_batch([record]),device))[0].cpu().numpy()
    pools=observed_pools(record,catalog)
    if embeddings is None: embeddings=encode_observed_catalog(model,catalog,[record],device)
    values=[]; ranks=[]; selected=[]
    k=int(settings['candidate_k'])
    if k<10: raise ValueError('candidate_k must be at least 10 for Top-10 evaluation')
    for i,pool in enumerate(pools):
        scores=(embeddings[pool]@query[i])/model.cfg.temperature if pool else np.empty(0)
        ranked=_rank(scores,pool) if pool else []
        values.append(dict(zip(pool,scores.tolist()))); ranks.append(ranked); selected.append(ranked[:k])
    n=len(pools); width=max(1,max(map(len,selected),default=0))
    unary=torch.zeros(n,width,device=device); valid=torch.zeros(n,width,dtype=torch.bool,device=device)
    fragment=torch.zeros(n,width,model.cfg.hidden,device=device); masses=torch.zeros(n,width,device=device)
    for i,row in enumerate(selected):
        if not row: continue
        unary[i,:len(row)]=torch.tensor([values[i][x] for x in row],device=device)
        valid[i,:len(row)]=True
        fragment[i,:len(row)]=torch.as_tensor(embeddings[row],device=device)
        masses[i,:len(row)]=torch.as_tensor(catalog.masses[row],device=device,dtype=torch.float32)
    c=spectrum_batch([record])['conditions'][0].to(device).expand(n,-1)
    return dict(pools=pools,unary_rank=ranks,unary_scores=values,selected=selected,candidate_k=k,
        unary=unary,valid=valid,fragment=fragment,masses=masses,conditions=c,
        peak_masses=torch.tensor([p[0] for p in record['peaks']],device=device))


@torch.no_grad()
def rank_prepared(model,prepared,settings):
    if not math.isfinite(float(settings['strength'])) or settings['strength']<0:
        raise ValueError('Joint strength must be finite and nonnegative')
    if int(settings['candidate_k'])!=prepared['candidate_k']:
        raise ValueError('Cannot reuse prepared retrieval with a different candidate_k')
    kwargs={x:settings[x] for x in ('strength','iterations','neighbors','support_k','damping')}
    logits=joint_update(prepared['unary'],prepared['fragment'],prepared['masses'],prepared['valid'],
        prepared['peak_masses'],prepared['conditions'],model.relation,**kwargs).cpu().numpy()
    joint=[]; score_maps=[]
    for i,row in enumerate(prepared['selected']):
        joint.append(_rank(logits[i,:len(row)],row) if row else [])
        score_maps.append(dict(zip(row,logits[i,:len(row)].tolist())))
    return dict(pools=prepared['pools'],unary_rank=prepared['unary_rank'],unary_scores=prepared['unary_scores'],
                joint_rank=joint,joint_scores=score_maps)


@torch.no_grad()
def rank_record(model,record,catalog,device,settings,embeddings=None,query=None):
    """Backward-compatible one-record entry point for deployment and tests."""
    return rank_prepared(model,prepare_record(model,record,catalog,device,settings,embeddings,query),settings)
