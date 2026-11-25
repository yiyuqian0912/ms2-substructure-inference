"""Held-out teacher-relation audit, deliberately unavailable on test DAGs."""
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from casmi_peaks.data import Catalog,PeakDataset,digest
from casmi_peaks.engine import device_context,move
from .engine import load_model
from .relations import RelationStore,relation_examples,TYPES


@torch.no_grad()
def audit_relations(checkpoint,data,relations,output,device_name='auto',split='val'):
    if split not in ('train','val'): raise ValueError('Test DAG auditing is forbidden')
    device,_,world=device_context(device_name)
    if world!=1: raise ValueError('Relation audit uses one process')
    model,ckpt=load_model(checkpoint,data,device); dataset=PeakDataset(data,['experimental'],split)
    catalog=Catalog(data); store=RelationStore(relations,dataset.meta)
    if store.meta!=ckpt['relation_signature']: raise ValueError('Relation audit/checkpoint identity differs')
    logits=[]; labels=[]; weights=[]; skipped=0
    for record in dataset:
        samples=relation_examples([record],catalog,store,random.Random(20261112+record['id']),16)
        if not samples: skipped+=1; continue
        ids=sorted({x for a,b,_,_,_ in samples for x in (a,b)}); lookup={k:i for i,k in enumerate(ids)}
        f=model.fragment(move(catalog.graphs(ids),device)); left,right=model.relation.project(f)
        a=torch.tensor([lookup[a] for a,_,_,_,_ in samples],device=device)
        b=torch.tensor([lookup[b] for _,b,_,_,_ in samples],device=device)
        ma=torch.tensor([catalog.masses[a] for a,_,_,_,_ in samples],device=device,dtype=torch.float32)
        mb=torch.tensor([catalog.masses[b] for _,b,_,_,_ in samples],device=device,dtype=torch.float32)
        c=torch.tensor([c for _,_,c,_,_ in samples],device=device)
        logits.append(model.relation.logits(left[a],right[b],ma,mb,c).cpu())
        labels.extend(y for _,_,_,y,_ in samples); weights.extend(w for _,_,_,_,w in samples)
    store.close(); result={}
    if logits:
        logits=torch.cat(logits); y=torch.tensor(labels); w=torch.tensor(weights); probabilities=logits.sigmoid()
        bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
        for i,name in enumerate(TYPES):
            positive=y[:,i].bool(); negative=~positive
            effective=w*torch.where(positive|(w<1),1.,.25)
            result[name]=dict(positive_pairs=int(positive.sum()),weak_negative_pairs=int(negative.sum()),
                weighted_bce=float((bce[:,i]*effective).sum()/effective.sum()),
                mean_positive_score=float(probabilities[positive,i].mean()) if positive.any() else None,
                mean_weak_negative_score=float(probabilities[negative,i].mean()) if negative.any() else None,
                weighted_average_precision=float(average_precision_score(y[:,i].numpy(),probabilities[:,i].numpy(),sample_weight=effective.numpy())) if positive.any() and negative.any() else None)
    packet=dict(split=split,checkpoint_sha256=digest(checkpoint),spectra=len(dataset),skipped_no_supported_pairs=skipped,
        sampled_pairs=len(labels),partial_data=dataset.meta['partial'],metrics=result,
        interpretation='Held-out AIMe relation agreement. Corruptions and missing edges are weak negatives; scores/AP do not measure chemical validity or exact fragmentation paths.')
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),sampled_pairs=len(labels))),flush=True); return packet
