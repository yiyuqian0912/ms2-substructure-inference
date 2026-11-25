"""Train/evaluate a peak-to-fragment student, with explicit coverage denominators."""
from collections import Counter,defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from .data import Catalog,PeakDataset,spectrum_batch,training_targets,signature,digest
from .model import ModelConfig,PeakAnnotator,retrieval_loss

KS=(1,3,5,10)


def identity(records):return records


def device_context(name='auto',threads=2):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA is unavailable')
        torch.cuda.set_device(local);device=torch.device('cuda',local)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        # Match the working H20 stable profile; no fused attention checkpoint path.
        for key,value in [('enable_flash_sdp',False),('enable_mem_efficient_sdp',False),('enable_math_sdp',True),('enable_cudnn_sdp',False)]:
            function=getattr(torch.backends.cuda,key,None)
            if function:function(value)
    else:device=torch.device(name)
    torch.set_num_threads(threads)
    if world>1:dist.init_process_group(backend='nccl' if device.type=='cuda' else 'gloo')
    return device,rank,world


def move(data,device):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}


def amp_context(device,mode):
    if mode not in ('none','bf16'):raise ValueError('amp must be none or bf16')
    if device.type=='cuda' and mode=='bf16':return torch.autocast('cuda',dtype=torch.bfloat16)
    return nullcontext()


def atomic_save(data,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(data,temporary);temporary.replace(path)


@torch.no_grad()
def encode_catalog(model,catalog,device,batch_size=256,mode='none'):
    model.eval();output=[]
    for start in range(0,len(catalog.labels),batch_size):
        ids=list(range(start,min(len(catalog.labels),start+batch_size)))
        with amp_context(device,mode):encoded=model.fragment(move(catalog.graphs(ids),device))
        output.append(encoded.cpu())
    return torch.cat(output).numpy()


def _rank(scores,ids):
    # Stable catalog-ID tie breaks make model and baselines reproducible.
    return np.asarray(ids)[np.lexsort((np.asarray(ids),-np.asarray(scores)))].tolist()


@torch.no_grad()
def evaluate_model(model,dataset,catalog,device,batch_size=8,workers=0,mode='none',catalog_batch=256):
    model.eval();rank=dist.get_rank() if dist.is_initialized() else 0;world=dist.get_world_size() if dist.is_initialized() else 1
    embeddings=encode_catalog(model,catalog,device,catalog_batch,mode)
    shard=Subset(dataset,list(range(rank,len(dataset),world)))
    loader=DataLoader(shard,batch_size=batch_size,collate_fn=identity,num_workers=workers)
    count=Counter();query=defaultdict(Counter)
    for records in loader:
        with amp_context(device,mode):queries=model.encode_spectrum(move(spectrum_batch(records),device)).cpu().numpy()
        for b,r in enumerate(records):
            count['spectra']+=1;signal_total=signal_annotated=0.
            for i,(mz,intensity) in enumerate(r['peaks']):
                count['total_peaks']+=1
                if r['precursor_peak'][i]:count['precursor_peaks']+=1;continue
                count['eligible_fragment_peaks']+=1;signal_total+=math.sqrt(intensity)
                targets=r['targets'][i]
                if not targets:count['unannotated_fragment_peaks']+=1;continue
                count['teacher_annotated_peaks']+=1;signal_annotated+=math.sqrt(intensity)
                positive={catalog.keys[k] for k,_ in targets if k in catalog.keys}
                ids=catalog.compatible(mz,r['precursor_type']);covered=positive.intersection(ids)
                count['candidate_count_sum']+=len(ids)
                count['ambiguous_teacher_peaks']+=len(targets)>1
                if covered:count['vocab_covered_peaks']+=1
                else:count['uncovered_teacher_peaks']+=1
                if positive and not covered:count['mass_gate_missed_known_label_peaks']+=1
                if covered and set(ids)-positive:count['discriminative_peaks']+=1
                q=query[str(r['qid'])];q['annotated']+=1
                if ids:
                    scores=embeddings[ids]@queries[b,i]
                    ranked=_rank(scores,ids)
                    baseline=sorted(ids,key=lambda k:(abs(catalog.masses[k]-mz),k))
                    frequent=sorted(ids,key=lambda k:(-catalog.freq[k],abs(catalog.masses[k]-mz),k))
                    if covered:
                        logits=torch.tensor(scores,dtype=torch.float64)/model.cfg.temperature
                        mask=torch.tensor([k in positive for k in ids])
                        count['set_nll_sum']+=float(torch.logsumexp(logits,0)-torch.logsumexp(logits[mask],0))
                else:ranked=[];baseline=[];frequent=[]
                for k in KS:
                    hit=bool(positive.intersection(ranked[:k]))
                    count[f'hit{k}']+=hit;q[f'hit{k}']+=hit
                    count[f'mass_hit{k}']+=bool(positive.intersection(baseline[:k]))
                    count[f'frequency_hit{k}']+=bool(positive.intersection(frequent[:k]))
                    if covered:count[f'covered_hit{k}']+=hit
                    if covered and set(ids)-positive:count[f'discriminative_hit{k}']+=hit
                    if len(targets)==1:count[f'singleton_hit{k}']+=hit
                if len(targets)==1:count['singleton_peaks']+=1
                ranks=[j+1 for j,k in enumerate(ranked) if k in positive]
                if ranks:count['reciprocal_rank_sum']+=1/min(ranks)
            count['teacher_signal_coverage_sum']+=signal_annotated/signal_total if signal_total else 0.
    if dist.is_initialized():
        all_stats=[None]*world;dist.all_gather_object(all_stats,(dict(count),{k:dict(v) for k,v in query.items()}))
        count=Counter();query=defaultdict(Counter)
        for c,qs in all_stats:
            count.update(c)
            for key,v in qs.items():query[key].update(v)
    annotated=count['teacher_annotated_peaks'];covered=count['vocab_covered_peaks'];eligible=count['eligible_fragment_peaks']
    ratio=lambda a,b:float(a/b) if b else None
    result={k:int(count[k]) for k in ['spectra','total_peaks','precursor_peaks','eligible_fragment_peaks','unannotated_fragment_peaks',
        'teacher_annotated_peaks','vocab_covered_peaks','uncovered_teacher_peaks','mass_gate_missed_known_label_peaks',
        'ambiguous_teacher_peaks','singleton_peaks','discriminative_peaks']}
    result.update(teacher_peak_coverage=ratio(annotated,eligible),vocabulary_coverage_of_annotated=ratio(covered,annotated),
        teacher_signal_coverage_spectrum_mean=ratio(count['teacher_signal_coverage_sum'],count['spectra']),
        mean_mass_compatible_candidates=ratio(count['candidate_count_sum'],annotated),
        set_nll_vocab_covered=ratio(count['set_nll_sum'],covered),teacher_mrr_all_annotated=ratio(count['reciprocal_rank_sum'],annotated))
    for k in KS:
        result[f'teacher_top{k}_all_annotated']=ratio(count[f'hit{k}'],annotated)
        result[f'teacher_top{k}_vocab_covered']=ratio(count[f'covered_hit{k}'],covered)
        result[f'teacher_top{k}_discriminative']=ratio(count[f'discriminative_hit{k}'],count['discriminative_peaks'])
        result[f'teacher_top{k}_singleton']=ratio(count[f'singleton_hit{k}'],count['singleton_peaks'])
        result[f'mass_only_top{k}_all_annotated']=ratio(count[f'mass_hit{k}'],annotated)
        result[f'training_frequency_top{k}_all_annotated']=ratio(count[f'frequency_hit{k}'],annotated)
        rates=[v[f'hit{k}']/v['annotated'] for v in query.values() if v['annotated']]
        result[f'teacher_top{k}_query_macro']=sum(rates)/len(rates) if rates else None
    result['queries_with_annotations']=len(query)
    result['interpretation']='Agreement with AIMe pseudo-label sets, not independently validated chemical accuracy. OOV labels count as failures in all_annotated metrics; unannotated peaks are reported as coverage, not negatives.'
    return result


def checkpoint_guard(checkpoint,cfg,meta,subset=None,resume=False,world=1):
    if checkpoint.get('task')!='peak_fragment_retrieval_v1':raise ValueError('This is not a peak-annotation checkpoint; old DAG-reconstruction weights are incompatible')
    if checkpoint['dataset_signature']!=signature(meta):raise ValueError('Checkpoint catalog/data signature differs')
    if checkpoint['model_config']!=cfg['model']:raise ValueError('Checkpoint model configuration differs')
    if resume:
        if checkpoint['config']!=cfg or checkpoint['subset_signature']!=subset:raise ValueError('Resume configuration/subset differs; use the original settings or a new output directory')
        if checkpoint['world_size']!=world:raise ValueError('Exact resume requires the original number of processes')


def train(config,data,output,device_name='auto',resume=None,initialize=None,limit=None,epochs_override=None):
    cfg=json.loads(Path(config).read_text());cfg['model']=asdict(ModelConfig(**cfg.get('model',{})))
    tc=cfg['training']
    if epochs_override is not None:tc['epochs']=epochs_override
    if resume and initialize:raise ValueError('Choose resume or initialize')
    if tc['epochs']<1 or tc['batch_size']<1 or tc.get('accumulation',1)<1 or tc['negatives']<1:raise ValueError('Invalid training configuration')
    if cfg['loss_mode'] not in ['soft','set']:raise ValueError('Unknown loss mode')
    device,rank,world=device_context(device_name,tc.get('cpu_threads',2))
    seed=cfg.get('seed',20261111);torch.manual_seed(seed+rank);random.seed(seed+rank)
    if device.type=='cuda':torch.cuda.manual_seed(seed+rank)
    training=PeakDataset(data,cfg['sources'],'train',limit);validation=PeakDataset(data,cfg['sources'],'val',limit)
    training.index=[r for r in training.index if r['annotated_peaks']>0]
    if not len(training) or not len(validation):raise ValueError('Empty training/validation subset')
    catalog=Catalog(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not resume and any((output/f).exists() for f in ['last.pt','best.pt']):raise ValueError('Output already has checkpoints; use --resume or a new --output')
    subset=hashlib.sha256(json.dumps([training.index,validation.index],sort_keys=True).encode()).hexdigest()
    model=PeakAnnotator(ModelConfig(**cfg['model'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc.get('weight_decay',.001),foreach=False)
    sampler=DistributedSampler(training,world,rank,shuffle=True,seed=seed) if world>1 else None
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(training,batch_size=tc['batch_size'],sampler=sampler,shuffle=sampler is None,
        collate_fn=identity,num_workers=tc.get('workers',0),generator=generator)
    accumulation=tc.get('accumulation',1);steps_per_epoch=math.ceil(len(loader)/accumulation)
    total_steps=tc['epochs']*steps_per_epoch;warmup=max(1,int(total_steps*tc.get('warmup_fraction',.03)))
    schedule=lambda step:max(.05,step/warmup) if step<warmup else .05+.95*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    start=0;best=-1.;bad=0;global_step=0
    if resume or initialize:
        ckpt=torch.load(resume or initialize,map_location='cpu',weights_only=True)
        checkpoint_guard(ckpt,cfg,training.meta,subset,resume=bool(resume),world=world)
        model.load_state_dict(ckpt['model'])
        if resume:
            optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
            start=ckpt['epoch']+1;best=ckpt['best_score'];bad=ckpt['bad_epochs'];global_step=ckpt['global_step']
            rng=ckpt['rng_states'][rank];torch.set_rng_state(rng['torch'])
            if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'],device)
            generator.set_state(ckpt['loader_generator'])
    if world>1:wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=False)
    else:wrapped=model
    if rank==0:
        (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        print(json.dumps(dict(task='peak_fragment_retrieval_v1',parameters=sum(p.numel() for p in model.parameters()),
            train_spectra=len(training),val_spectra=len(validation),vocabulary=len(catalog.labels),world_size=world,
            input='spectrum and conditions only',partial_data=training.meta['partial'],loss_mode=cfg['loss_mode'],amp=tc.get('amp','none'))),flush=True)
    for epoch in range(start,tc['epochs']):
        clock=time.monotonic();wrapped.train();optimizer.zero_grad(set_to_none=True)
        if sampler:sampler.set_epoch(epoch)
        # Epoch-deterministic data order even when a single-process run resumes.
        if sampler is None:generator.manual_seed(seed+epoch)
        total_loss=0.;total_peaks=0
        for step,records in enumerate(loader):
            targets=training_targets(records,catalog,random.Random(seed+epoch*1000003+rank*10007+step),tc['max_target_peaks'],tc['negatives'])
            if targets is None:raise ValueError('Training batch has no catalog-covered pseudo-labels')
            batch=move(spectrum_batch(records),device);graphs=move(catalog.graphs(targets['ids']),device)
            target_device=move({k:v for k,v in targets.items() if k!='ids'},device)
            boundary=(step+1)%accumulation==0 or step+1==len(loader)
            context=wrapped.no_sync() if world>1 and not boundary else nullcontext()
            count=len(targets['positions']);global_count=torch.tensor(float(count),device=device)
            if world>1:dist.all_reduce(global_count)
            actual_accumulation=min(accumulation,len(loader)-(step//accumulation)*accumulation)
            with context:
                with amp_context(device,tc.get('amp','none')):queries,fragments=wrapped(batch,graphs)
                loss=retrieval_loss(queries,fragments,target_device,model.cfg.temperature,cfg['loss_mode'])
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
                (loss*(world*count/float(global_count))/actual_accumulation).backward()
            total_loss+=float(loss.detach())*count;total_peaks+=count
            if boundary:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get('gradient_clip',1.),foreach=False,error_if_nonfinite=True)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
                if rank==0 and global_step%tc.get('log_every',50)==0:
                    print(json.dumps(dict(epoch=epoch,step=global_step,loss=float(loss.detach()),gradient_norm=float(norm),lr=scheduler.get_last_lr()[0])),flush=True)
        totals=torch.tensor([total_loss,total_peaks],dtype=torch.float64,device=device)
        if world>1:dist.all_reduce(totals)
        metrics=evaluate_model(model,validation,catalog,device,tc.get('eval_batch_size',8),tc.get('workers',0),tc.get('amp','none'),tc.get('catalog_batch',256))
        score=metrics['teacher_top1_all_annotated']
        if score is None:raise ValueError('Validation has no annotated fragment peaks')
        improved=score>best
        if improved:best=score;bad=0
        else:bad+=1
        rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
        rng_states=[rng]
        if world>1:rng_states=[None]*world;dist.all_gather_object(rng_states,rng)
        if rank==0:
            payload=dict(task='peak_fragment_retrieval_v1',dataset_signature=signature(training.meta),subset_signature=subset,
                model_config=cfg['model'],config=cfg,epoch=epoch,global_step=global_step,best_score=best,bad_epochs=bad,
                world_size=world,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng_states=rng_states,loader_generator=generator.get_state(),selection_metric='validation teacher_top1_all_annotated',
                input_contract='No DAG, precursor structure, matching arrays or target flags enter the spectrum encoder')
            atomic_save(payload,output/'last.pt')
            if improved:atomic_save(payload,output/'best.pt')
            row=dict(epoch=epoch,global_step=global_step,train_loss=float(totals[0]/totals[1]),
                validation=metrics,best_score=best,bad_epochs=bad,seconds=time.monotonic()-clock)
            with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row,allow_nan=False),flush=True)
        if world>1:dist.barrier()
        if bad>=tc.get('patience',12):break
    if world>1:dist.destroy_process_group()


def load_model(checkpoint,data,device):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    meta=json.loads((Path(data)/'meta.json').read_text())
    checkpoint_guard(ckpt,{'model':ckpt['model_config']},meta)
    model=PeakAnnotator(ModelConfig(**ckpt['model_config'])).to(device);model.load_state_dict(ckpt['model']);model.eval()
    return model,ckpt


def evaluate(checkpoint,data,output,split='test',sources=('experimental',),device_name='auto',limit=None):
    device,rank,world=device_context(device_name)
    dataset=PeakDataset(data,list(sources),split,limit)
    if not len(dataset):raise ValueError('Empty evaluation split')
    catalog=Catalog(data);model,ckpt=load_model(checkpoint,data,device)
    result=evaluate_model(model,dataset,catalog,device)
    packet=dict(split=split,sources=list(sources),checkpoint_epoch=ckpt['epoch'],checkpoint_sha256=digest(checkpoint),
                partial_data=dataset.meta['partial'],
                dataset_signature=ckpt['dataset_signature'],metrics=result)
    if rank==0:
        Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
        print(json.dumps(packet,indent=2,allow_nan=False),flush=True)
    if world>1:dist.destroy_process_group()
    return packet
