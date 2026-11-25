"""Annotate an experimental spectrum without accepting a precursor structure/DAG."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .chem import ion_sign
from .data import Catalog,signature,spectrum_batch,tolerance,digest
from .engine import device_context,load_model,move,_rank


def read_spectrum(content):
    if not isinstance(content,dict):raise ValueError('Expected one JSON object')
    if 'experimental_ms2' in content:
        peaks=content['experimental_ms2'];md=content['experiment_metadata']
        precursor=md['precursor_mass'];energy=md['collision_energy_nce'];ion=md['precursor_type']
    else:
        peaks=content['peaks'];md=content.get('conditions',content)
        precursor=md['precursor_mz'];energy=md['collision_energy_nce'];ion=md['precursor_type']
    ion_sign(ion);precursor=float(precursor);energy=float(energy)
    if not math.isfinite(precursor) or precursor<=0 or not math.isfinite(energy):raise ValueError('Invalid precursor mass/energy')
    clean=[]
    for mz,intensity in peaks:
        mz=float(mz);intensity=float(intensity)
        if not math.isfinite(mz) or not math.isfinite(intensity) or mz<=0 or intensity<0:raise ValueError('Invalid peak')
        if intensity>0:clean.append([mz,intensity])
    clean.sort(key=lambda p:p[0])
    if not clean:raise ValueError('No positive-intensity peaks')
    # Whitelist is intentional: DAG, true molecule, teacher arrays and IDs are ignored.
    return dict(peaks=clean,precursor_mz=precursor,collision_energy_nce=energy,precursor_type=ion)


@torch.no_grad()
def infer(checkpoint,catalog_path,input_path,output,topk=10,device_name='auto'):
    if topk<1:raise ValueError('topk must be positive')
    device,rank,world=device_context(device_name)
    if world!=1:raise ValueError('Inference is a single-process command')
    record=read_spectrum(json.loads(Path(input_path).read_text()))
    catalog=Catalog(catalog_path);model,ckpt=load_model(checkpoint,catalog_path,device)
    queries=model.encode_spectrum(move(spectrum_batch([record]),device))[0].cpu().numpy()
    candidates=[]
    for mz,_ in record['peaks']:
        if abs(mz-record['precursor_mz'])<=tolerance(mz,catalog.cfg):candidates.append([])
        else:candidates.append(catalog.compatible(mz,record['precursor_type']))
    ids=sorted({i for row in candidates for i in row});embeddings=[]
    for start in range(0,len(ids),256):embeddings.append(model.fragment(move(catalog.graphs(ids[start:start+256]),device)).cpu())
    lookup={i:j for j,i in enumerate(ids)}
    embeddings=torch.cat(embeddings).numpy() if embeddings else np.empty((0,model.cfg.hidden))
    annotated=[]
    for i,((mz,intensity),pool) in enumerate(zip(record['peaks'],candidates)):
        precursor_peak=abs(mz-record['precursor_mz'])<=tolerance(mz,catalog.cfg)
        ranked=[];scores={}
        if pool:
            values=embeddings[[lookup[k] for k in pool]]@queries[i]
            ranked=_rank(values,pool)[:topk];scores=dict(zip(pool,values.tolist()))
        alternatives=[]
        for k in ranked:
            label=catalog.labels[k]
            alternatives.append(dict(catalog_id=k,exported_structure_smiles=label['smiles'],
                reported_ion_formula=label['formula'],precursor_type=label['ion'],
                catalog_mz=label['mz'],mass_error_da=abs(label['mz']-mz),
                retrieval_score=scores[k],formula_smiles_consistent=label['formula_smiles_consistent']))
        annotated.append(dict(peak_index=i,mz=mz,intensity=intensity,mass_compatible_candidates=len(pool),
            status='precursor_peak_not_annotated' if precursor_peak else 'ranked_unconfirmed_candidates' if pool else 'no_mass_compatible_training_catalog_candidate',
            candidates=alternatives))
    result=dict(task='peak_fragment_retrieval_v1',checkpoint_epoch=ckpt['epoch'],checkpoint_sha256=digest(checkpoint),
        partial_catalog=ckpt['dataset_signature']['partial'],
        input_conditions={k:v for k,v in record.items() if k!='peaks'},catalog_size=len(catalog.labels),
        interpretation='DAG-free retrieval from a train-only fragment catalog. Scores are uncalibrated similarities, not chemical confidence. A mass-compatible catalog candidate does not prove an OOV peak is explained. SMILES describe the AIMe-exported graph; reported ion composition is separate.',
        peaks=annotated)
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),peaks=len(annotated),with_candidates=sum(bool(p['candidates']) for p in annotated))),flush=True)
    return result
