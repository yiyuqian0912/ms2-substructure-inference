"""DDP training for mass-aware unary retrieval plus typed DAG relations."""
from dataclasses import asdict
from contextlib import nullcontext,ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from casmi_peaks.data import Catalog,PeakDataset,spectrum_batch,signature,digest
from casmi_peaks.engine import device_context,move,atomic_save,identity
from . import TASK,IMPLEMENTATION_REVISION
from .model import JointConfig,JointAnnotator
from .relations import RelationStore,make_training
from .evaluate import evaluate_model


JOINT_DEFAULTS=dict(candidate_k=32,strength=2.,iterations=2,neighbors=6,support_k=16,damping=.5)


def read_config(path):
    cfg=json.loads(Path(path).read_text())
    cfg['model']=asdict(JointConfig(**cfg.get('model',{})))
    cfg['joint']={**JOINT_DEFAULTS,**cfg.get('joint',{})}
    t=cfg['training']; j=cfg['joint']
    t.setdefault('context_peaks',8)
    if cfg['loss_mode'] not in ('set','soft') or not set(cfg['sources'])<={'predicted','experimental'}: raise ValueError('Invalid source/loss')
    if min(t['epochs'],t['batch_size'],t['max_target_peaks'],t['negatives'],t['relation_pairs'],t.get('accumulation',1))<1:
        raise ValueError('Invalid training configuration')
    if t.get('amp','none')!='none': raise ValueError('V2 release uses the verified FP32/math profile')
    if t['context_peaks']<0: raise ValueError('context_peaks must be nonnegative')
    if t['relation_weight']<0: raise ValueError('Relation auxiliary weight must be nonnegative')
    for name,default,positive in [('learning_rate',0.,True),('weight_decay',.001,False),('relation_weight',0.,False),
                                ('unary_weight',.2,False),('gradient_clip',1.,True)]:
        value=t.get(name,default)
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or (value<=0 if positive else value<0):
            raise ValueError(f'Invalid {name}')
    if not 0<=t.get('warmup_fraction',.03)<=1: raise ValueError('Invalid warmup_fraction')
    if j['candidate_k']<10 or j['neighbors']<1 or j['support_k']<1 or j['iterations']<0 or j['strength']<0 or not 0<j['damping']<=1:
        raise ValueError('Invalid joint configuration')
    if set(j)!=set(JOINT_DEFAULTS): raise ValueError('Unknown joint settings')
    return cfg


def subset_signature(training,validation):
    return hashlib.sha256(json.dumps([training.index,validation.index,getattr(training,'unlabelled_index',[])],sort_keys=True).encode()).hexdigest()


def checkpoint_guard(checkpoint,cfg,meta,relation_meta=None,subset=None,resume=False,world=1):
    if checkpoint.get('task')!=TASK: raise ValueError('V2 requires V2 weights; V1/old reconstruction weights cannot initialize this architecture')
    if checkpoint['dataset_signature']!=signature(meta): raise ValueError('Checkpoint catalog/data signature differs')
    if checkpoint['model_config']!=cfg['model']: raise ValueError('Checkpoint model configuration differs')
    if relation_meta is not None and checkpoint['relation_signature']!=relation_meta: raise ValueError('Relation signature differs')
    if resume:
        if checkpoint.get('implementation_revision')!=IMPLEMENTATION_REVISION: raise ValueError('Resume implementation revision differs; start an audited run')
        if checkpoint['config']!=cfg or checkpoint['subset_signature']!=subset: raise ValueError('Resume configuration/subset differs')
        if checkpoint['world_size']!=world: raise ValueError('Exact resume requires the same world size')


def train(config,data,relations,output,device_name='auto',resume=None,initialize=None,limit=None):
    with ExitStack() as resources:
        return _train(config,data,relations,output,device_name,resume,initialize,limit,resources)


def _train(config,data,relations,output,device_name,resume,initialize,limit,resources):
    cfg=read_config(config); tc=cfg['training']
    if resume and initialize: raise ValueError('Choose resume or initialize')
    device,rank,world=device_context(device_name,tc.get('cpu_threads',2))
    seed=cfg.get('seed',20261112); torch.manual_seed(seed+rank); random.seed(seed+rank)
    if device.type=='cuda': torch.cuda.manual_seed(seed+rank)
    training=PeakDataset(data,cfg['sources'],'train',limit); validation=PeakDataset(data,cfg['sources'],'val',limit)
    resources.callback(training.close); resources.callback(validation.close)
    unlabelled_records=[training[i] for i,row in enumerate(training.index) if row['annotated_peaks']==0]
    training.unlabelled_index=[row for row in training.index if row['annotated_peaks']==0]
    training.index=[r for r in training.index if r['annotated_peaks']>0]
    if not len(training) or not len(validation): raise ValueError('Empty training/validation subset')
    store=RelationStore(relations,training.meta); catalog=Catalog(data)
    resources.callback(store.close)
    relation_only=[record for record in unlabelled_records if len(store.get(record).keys)>=2] if tc['relation_weight']>0 else []
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if not resume and any((output/n).exists() for n in ('last.pt','best.pt')): raise ValueError('Output exists; resume or choose a new run root')
    subset=subset_signature(training,validation); model=JointAnnotator(JointConfig(**cfg['model'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc.get('weight_decay',.001),foreach=False)
    sampler=DistributedSampler(training,world,rank,shuffle=True,seed=seed) if world>1 else None
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(training,batch_size=tc['batch_size'],sampler=sampler,shuffle=sampler is None,collate_fn=identity,
                      num_workers=tc.get('workers',0),generator=generator)
    accumulation=tc.get('accumulation',1); total=tc['epochs']*math.ceil(len(loader)/accumulation)
    warmup=max(1,int(total*tc.get('warmup_fraction',.03)))
    schedule=lambda step:max(.05,step/warmup) if step<warmup else .05+.95*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    start=0; best=-1.; bad=0; global_step=0
    initialization=None
    if resume or initialize:
        ckpt=torch.load(resume or initialize,map_location='cpu',weights_only=True)
        checkpoint_guard(ckpt,cfg,training.meta,store.meta,subset,bool(resume),world)
        model.load_state_dict(ckpt['model'])
        initialization=ckpt.get('initialization') if resume else dict(checkpoint_sha256=digest(initialize),
            partial=bool(ckpt.get('limited_training') or (ckpt.get('initialization') or {}).get('partial') or ckpt['dataset_signature']['partial']),
            implementation_revision=ckpt.get('implementation_revision',1))
        if resume:
            optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
            start=ckpt['epoch']+1; best=ckpt['best_score']; bad=ckpt['bad_epochs']; global_step=ckpt['global_step']
            torch.set_rng_state(ckpt['rng_states'][rank]['torch'])
            if device.type=='cuda': torch.cuda.set_rng_state(ckpt['rng_states'][rank]['cuda'],device)
            generator.set_state(ckpt['loader_generator'])
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=False) if world>1 else model
    train_joint={k:v for k,v in cfg['joint'].items() if k!='candidate_k'}
    if rank==0:
        (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        print(json.dumps(dict(task=TASK,parameters=sum(p.numel() for p in model.parameters()),train_spectra=len(training),
            relation_only_spectra=len(relation_only),val_spectra=len(validation),world_size=world,
            partial_data=training.meta['partial'] or limit is not None or bool(initialization and initialization['partial']),input='spectrum and conditions only',
            relation_types=store.meta['types'],training_candidates='all positives plus sampled same-mass negatives; no evaluation target injection')),flush=True)
    for epoch in range(start,tc['epochs']):
        clock=time.monotonic(); wrapped.train(); optimizer.zero_grad(set_to_none=True)
        if sampler: sampler.set_epoch(epoch)
        else: generator.manual_seed(seed+epoch)
        totals=torch.zeros(6,dtype=torch.float64,device=device)
        # Normalize over the entire accumulation window, not the mean of
        # microbatch means, which biases unequal annotated-peak counts.
        iterator=iter(loader); step=0
        while step<len(loader):
          window=[]
          for local_step in range(min(accumulation,len(loader)-step)):
            records=next(iterator); absolute_step=step+local_step
            rng=random.Random(seed+epoch*1000003+rank*10007+absolute_step)
            replay_index=absolute_step*world+rank
            relation_records=records+([relation_only[replay_index]] if replay_index<len(relation_only) else [])
            if tc['relation_weight']==0: relation_records=[]
            targets,relation=make_training(records,catalog,store,rng,tc['max_target_peaks'],tc['negatives'],tc['relation_pairs'],cfg['loss_mode'],tc['context_peaks'],relation_records)
            effective=float((relation['weights'][:,None]*relation['head_weights']).sum()/4)
            window.append((records,targets,relation,int(targets['supervised'].sum()),effective))
          global_counts=torch.tensor([sum(x[3] for x in window),sum(x[4] for x in window)],device=device)
          if world>1: dist.all_reduce(global_counts)
          for local_step,(records,targets,relation,peak_count,relation_count) in enumerate(window):
            graphs=move(catalog.graphs(targets['ids']),device)
            targets=move({k:v for k,v in targets.items() if k!='ids'},device); relation=move(relation,device)
            counts=torch.tensor([peak_count,relation_count],device=device)
            boundary=local_step+1==len(window)
            context=wrapped.no_sync() if world>1 and not boundary else nullcontext()
            with context:
                losses=wrapped(move(spectrum_batch(records),device),graphs,targets,relation,train_joint)
                loss=(losses['peak_loss']+tc.get('unary_weight',.2)*losses['unary_loss'])*world*counts[0]/global_counts[0]
                loss=loss+tc['relation_weight']*losses['relation_loss']*world*counts[1]/global_counts[1].clamp_min(1e-12)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch} step {step}')
                loss.backward()
            totals+=torch.tensor([float(losses['peak_loss'].detach())*float(counts[0]),float(counts[0]),
                float(losses['relation_loss'].detach())*float(counts[1]),float(counts[1]),
                float(losses['unary_loss'].detach())*float(counts[0]),1.],dtype=torch.float64,device=device)
            if boundary:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get('gradient_clip',1.),foreach=False,error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1
                if rank==0 and global_step%tc.get('log_every',20)==0:
                    print(json.dumps(dict(epoch=epoch,step=global_step,loss=float(loss.detach()),gradient_norm=float(norm),lr=scheduler.get_last_lr()[0])),flush=True)
          step+=len(window)
        if world>1: dist.all_reduce(totals)
        metrics=evaluate_model(model,validation,catalog,device,cfg['joint'],batch_size=tc.get('eval_batch_size',8),catalog_batch=tc.get('catalog_batch',256))
        score=metrics['joint_by_strength'][f"{cfg['joint']['strength']:g}"]['teacher_top1_all_annotated']
        if score is None: raise ValueError('No annotated validation peaks')
        improved=score>best
        if improved: best=score; bad=0
        else: bad+=1
        rng_state={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
        states=[rng_state]
        if world>1: states=[None]*world; dist.all_gather_object(states,rng_state)
        if rank==0:
            payload=dict(task=TASK,dataset_signature=signature(training.meta),relation_signature=store.meta,subset_signature=subset,
                implementation_revision=IMPLEMENTATION_REVISION,initialization=initialization,relation_only_ids=[r['id'] for r in relation_only],
                model_config=cfg['model'],config=cfg,limited_training=limit is not None or bool(initialization and initialization['partial']),epoch=epoch,global_step=global_step,best_score=best,bad_epochs=bad,
                world_size=world,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng_states=states,loader_generator=generator.get_state(),selection_metric='validation joint teacher_top1_all_annotated')
            atomic_save(payload,output/'last.pt')
            if improved: atomic_save(payload,output/'best.pt')
            row=dict(epoch=epoch,global_step=global_step,train_peak_loss=float(totals[0]/totals[1]),
                train_unary_loss=float(totals[4]/totals[1]),train_relation_loss=float(totals[2]/totals[3].clamp_min(1.)),
                validation=metrics,best_score=best,bad_epochs=bad,seconds=time.monotonic()-clock)
            with (output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row,allow_nan=False),flush=True)
        if world>1: dist.barrier()
        if bad>=tc.get('patience',12): break
    store.close()
    if world>1: dist.destroy_process_group()


def load_model(checkpoint,data,device):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    meta=json.loads((Path(data)/'meta.json').read_text())
    checkpoint_guard(ckpt,{'model':ckpt['model_config']},meta)
    model=JointAnnotator(JointConfig(**ckpt['model_config'])).to(device)
    model.load_state_dict(ckpt['model']); model.eval(); return model,ckpt
