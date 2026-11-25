"""Full-pool unary and fixed-shortlist joint evaluation with explicit denominators."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import math
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader,Subset
from casmi_peaks.data import Catalog,PeakDataset,spectrum_batch,digest
from casmi_peaks.engine import move,identity,device_context
from .predict import encode_observed_catalog,prepare_record,rank_prepared

KS=(1,3,5,10)


def _summarize(count,queries):
    ratio=lambda a,b:float(a/b) if b else None
    n=count['annotated']; covered=count['covered']
    result={}
    for k in KS:
        result[f'teacher_top{k}_all_annotated']=ratio(count[f'hit{k}'],n)
        result[f'teacher_top{k}_vocab_covered']=ratio(count[f'hit{k}'],covered)
        result[f'teacher_top{k}_singleton']=ratio(count[f'singleton_hit{k}'],count['singleton'])
        result[f'teacher_top{k}_discriminative']=ratio(count[f'discriminative_hit{k}'],count['discriminative'])
        result[f'teacher_top{k}_query_macro']=sum(q[f'hit{k}']/q['annotated'] for q in queries.values())/len(queries) if queries else None
    return result


@torch.no_grad()
def evaluate_model(model,dataset,catalog,device,settings,strengths=None,batch_size=8,catalog_batch=256):
    model.eval(); strengths=sorted(set(strengths if strengths is not None else [settings['strength']]))
    if not strengths or any(not math.isfinite(float(s)) or s<0 for s in strengths):
        raise ValueError('Joint strengths must be finite, nonnegative and nonempty')
    rank=dist.get_rank() if dist.is_initialized() else 0; world=dist.get_world_size() if dist.is_initialized() else 1
    shard=Subset(dataset,list(range(rank,len(dataset),world)))
    embeddings=encode_observed_catalog(model,catalog,shard,device,catalog_batch)
    names=['unary']+[f'joint_{v:g}' for v in strengths]
    stats={name:Counter() for name in names}; qs={name:defaultdict(Counter) for name in names}; coverage=Counter(encoded_labels_sum=len(embeddings.ids))
    repairs={name:Counter() for name in names if name!='unary'}
    loader=DataLoader(shard,batch_size=batch_size,collate_fn=identity)
    for records in loader:
        encoded=model.encode_spectrum(move(spectrum_batch(records),device)).cpu().numpy()
        for b,record in enumerate(records):
            prepared=prepare_record(model,record,catalog,device,settings,embeddings,encoded[b])
            outputs={}
            for strength in strengths:
                outputs[f'joint_{strength:g}']=rank_prepared(model,prepared,{**settings,'strength':strength})
            base=next(iter(outputs.values())); coverage['spectra']+=1
            total_signal=annotated_signal=0.
            for i,(mz,intensity) in enumerate(record['peaks']):
                coverage['total_peaks']+=1
                if record['precursor_peak'][i]: coverage['precursor_peaks']+=1; continue
                coverage['eligible_fragment_peaks']+=1; total_signal+=math.sqrt(intensity)
                target=record['targets'][i]
                if not target: coverage['unannotated_fragment_peaks']+=1; continue
                coverage['teacher_annotated_peaks']+=1; annotated_signal+=math.sqrt(intensity)
                positive={catalog.keys[k] for k,_ in target if k in catalog.keys}
                pool=base['pools'][i]; covered=bool(positive.intersection(pool)); singleton=len(target)==1
                discriminative=covered and bool(set(pool)-positive)
                coverage['vocab_covered_peaks']+=covered; coverage['uncovered_teacher_peaks']+=not covered
                coverage['mass_gate_missed_known_label_peaks']+=bool(positive) and not covered
                coverage['ambiguous_teacher_peaks']+=not singleton; coverage['singleton_peaks']+=singleton
                coverage['discriminative_peaks']+=discriminative; coverage['candidate_count_sum']+=len(pool)
                unary=base['unary_rank'][i]
                for k in (16,32,64,128): coverage[f'recall{k}']+=bool(positive.intersection(unary[:k]))
                coverage['shortlist_covered_peaks']+=bool(positive.intersection(unary[:settings['candidate_k']]))
                mass=sorted(pool,key=lambda k:(abs(catalog.masses[k]-mz),k))
                frequency=sorted(pool,key=lambda k:(-catalog.freq[k],abs(catalog.masses[k]-mz),k))
                for k in KS:
                    coverage[f'mass_hit{k}']+=bool(positive.intersection(mass[:k]))
                    coverage[f'frequency_hit{k}']+=bool(positive.intersection(frequency[:k]))
                rankings={'unary':unary,**{name:o['joint_rank'][i] for name,o in outputs.items()}}
                for name,ranking in rankings.items():
                    count=stats[name]; query=qs[name][str(record['qid'])]
                    count['annotated']+=1; count['covered']+=covered; count['singleton']+=singleton
                    count['discriminative']+=discriminative; query['annotated']+=1
                    for k in KS:
                        hit=bool(positive.intersection(ranking[:k])); count[f'hit{k}']+=hit; query[f'hit{k}']+=hit
                        count[f'singleton_hit{k}']+=hit and singleton; count[f'discriminative_hit{k}']+=hit and discriminative
                    if name!='unary':
                        u=bool(positive.intersection(unary[:1])); j=bool(positive.intersection(ranking[:1]))
                        repairs[name]['corrected_top1']+=j and not u; repairs[name]['damaged_top1']+=u and not j
            coverage['signal_coverage_sum']+=annotated_signal/total_signal if total_signal else 0.
    payload=(coverage,stats,{k:dict(v) for k,v in qs.items()},repairs)
    if dist.is_initialized():
        gathered=[None]*world; dist.all_gather_object(gathered,payload)
    else: gathered=[payload]
    coverage=Counter(); stats={name:Counter() for name in names}; qs={name:defaultdict(Counter) for name in names}
    repairs={name:Counter() for name in names if name!='unary'}
    for c,s,q,r in gathered:
        coverage.update(c)
        for name in names:
            stats[name].update(s[name])
            for key,value in q[name].items(): qs[name][key].update(value)
        for name in repairs: repairs[name].update(r[name])
    ratio=lambda a,b:float(a/b) if b else None
    common={k:int(coverage[k]) for k in ('spectra','total_peaks','precursor_peaks','eligible_fragment_peaks',
        'unannotated_fragment_peaks','teacher_annotated_peaks','vocab_covered_peaks','uncovered_teacher_peaks',
        'mass_gate_missed_known_label_peaks','ambiguous_teacher_peaks','singleton_peaks','discriminative_peaks','shortlist_covered_peaks')}
    common.update(teacher_peak_coverage=ratio(coverage['teacher_annotated_peaks'],coverage['eligible_fragment_peaks']),
        vocabulary_coverage_of_annotated=ratio(coverage['vocab_covered_peaks'],coverage['teacher_annotated_peaks']),
        teacher_signal_coverage_spectrum_mean=ratio(coverage['signal_coverage_sum'],coverage['spectra']),
        mean_mass_compatible_candidates=ratio(coverage['candidate_count_sum'],coverage['teacher_annotated_peaks']),
        queries_with_annotations=len(qs['unary']))
    for k in (16,32,64,128): common[f'unary_candidate_recall_at_{k}']=ratio(coverage[f'recall{k}'],coverage['teacher_annotated_peaks'])
    common['joint_shortlist_oracle_top1_all_annotated']=ratio(coverage['shortlist_covered_peaks'],coverage['teacher_annotated_peaks'])
    for k in KS:
        common[f'mass_only_top{k}_all_annotated']=ratio(coverage[f'mass_hit{k}'],coverage['teacher_annotated_peaks'])
        common[f'training_frequency_top{k}_all_annotated']=ratio(coverage[f'frequency_hit{k}'],coverage['teacher_annotated_peaks'])
    return dict(coverage=common,unary=_summarize(stats['unary'],qs['unary']),
        joint_by_strength={f'{s:g}':{**_summarize(stats[f'joint_{s:g}'],qs[f'joint_{s:g}']),**dict(repairs[f'joint_{s:g}'])} for s in strengths},
        per_query={name:{qid:dict(values) for qid,values in query.items()} for name,query in qs.items()},
        settings=settings,evaluation_encoding=dict(strategy='rank-local union of all observed non-precursor mass-compatible candidates',
            local_encoded_labels=len(embeddings.ids),summed_encoded_labels=int(coverage['encoded_labels_sum']),
            world_size=world,catalog_labels=len(catalog.labels),unary_preparations_per_spectrum=1),interpretation='AIMe pseudo-label agreement, not chemical accuracy. Joint uses a fixed unary shortlist without teacher injection. Unannotated peaks provide context, not negatives.')


def load_calibration(path,checkpoint,settings,dataset_signature=None):
    packet=json.loads(Path(path).read_text())
    if packet.get('split')!='val' or packet.get('checkpoint_sha256')!=digest(checkpoint):
        raise ValueError('Calibration checkpoint/split mismatch')
    if packet.get('base_settings')!=settings: raise ValueError('Calibration inference settings mismatch')
    if packet.get('calibration_schema_version')!=2:
        raise ValueError('Old/unversioned calibration; rerun validation calibration')
    if dataset_signature is None:
        dataset_signature=torch.load(checkpoint,map_location='cpu',weights_only=True)['dataset_signature']
    if packet.get('dataset_signature')!=dataset_signature:
        raise ValueError('Calibration dataset signature mismatch')
    grid=packet.get('strength_grid'); selected=packet.get('selected_strength')
    numeric=lambda v:isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
    if not isinstance(grid,list) or not grid or not all(numeric(v) for v in grid) or not numeric(selected):
        raise ValueError('Calibration strengths must be finite and nonnegative')
    if selected not in grid: raise ValueError('Selected strength was not evaluated in calibration grid')
    if any(f'{v:g}' not in packet.get('metrics',{}).get('joint_by_strength',{}) for v in grid):
        raise ValueError('Calibration grid results are missing')
    return {**settings,'strength':float(selected)}


def evaluate(checkpoint,data,output,split='test',device_name='auto',calibration=None,strength=None,limit=None,source='experimental'):
    from .engine import load_model
    device,rank,world=device_context(device_name)
    model,ckpt=load_model(checkpoint,data,device); settings=ckpt['config']['joint']
    if calibration: settings=load_calibration(calibration,checkpoint,settings,ckpt['dataset_signature'])
    if strength is not None: settings={**settings,'strength':float(strength)}
    if source not in ('predicted','experimental'):raise ValueError('Invalid spectrum source')
    dataset=PeakDataset(data,[source],split,limit)
    if not len(dataset): raise ValueError('Empty evaluation set')
    result=evaluate_model(model,dataset,Catalog(data),device,settings)
    packet=dict(task='peak_fragment_joint_v2',split=split,sources=[source],checkpoint_epoch=ckpt['epoch'],checkpoint_sha256=digest(checkpoint),
        partial_data=dataset.meta['partial'] or limit is not None or ckpt.get('limited_training',False),
        limited_evaluation=limit is not None,limited_training=ckpt.get('limited_training',False),
        dataset_signature=ckpt['dataset_signature'],metrics=result)
    if rank==0:
        Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
        print(json.dumps(packet,indent=2,allow_nan=False),flush=True)
    if world>1: dist.destroy_process_group()
    return packet


def calibrate(checkpoint,data,output,device_name='auto',strengths=None):
    from .engine import load_model
    device,rank,world=device_context(device_name)
    model,ckpt=load_model(checkpoint,data,device); settings=ckpt['config']['joint']
    strengths=tuple(strengths if strengths is not None else ckpt['config'].get('calibration_strengths',(0.,.5,1.,2.,4.)))
    dataset=PeakDataset(data,['experimental'],'val')
    if not len(dataset): raise ValueError('Empty calibration validation set')
    result=evaluate_model(model,dataset,Catalog(data),device,settings,strengths=strengths)
    values=[result['joint_by_strength'][f'{s:g}']['teacher_top1_all_annotated'] for s in strengths]
    if any(v is None or not math.isfinite(v) for v in values): raise ValueError('Calibration has no finite annotated validation score')
    chosen=max(sorted(strengths),key=lambda s:result['joint_by_strength'][f'{s:g}']['teacher_top1_all_annotated'])
    packet=dict(calibration_schema_version=2,split='val',checkpoint_sha256=digest(checkpoint),
                dataset_signature=ckpt['dataset_signature'],strength_grid=sorted(set(strengths)),
                partial_data=dataset.meta['partial'] or ckpt.get('limited_training',False),
                base_settings=settings,selected_strength=chosen,
                selection_metric='validation teacher_top1_all_annotated; ties choose smaller strength',metrics=result)
    if rank==0:
        Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(calibration=str(output),selected_strength=chosen)),flush=True)
    if world>1: dist.destroy_process_group()
    return packet
