from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
import json
import hashlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler,Subset

from .data import SpectrumDAGDataset,collate,visibility,to_device
from .flow import losses,cosine_per_graph,segment_sum
from .model import ModelConfig,SpectrumDAGModel,normalized_model_config
from .runtime import native_diagnostics,configure_cuda,optimizer_options


def environment(device_name='auto',backend=None):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    device_name=('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name
    if device_name=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(local);device=torch.device('cuda',local)
    else:
        device=torch.device(device_name)
    if world>1:
        dist.init_process_group(backend=backend or ('nccl' if device.type=='cuda' else 'gloo'))
    return device,rank,world


def worker_init(worker_id):
    torch.set_num_threads(1)


def loader(dataset,batch_size,workers,device,sampler=None,shuffle=False):
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,shuffle=shuffle,
                      num_workers=workers,collate_fn=collate,pin_memory=device.type=='cuda',
                      persistent_workers=workers>0,worker_init_fn=worker_init,
                      multiprocessing_context='spawn' if workers else None,
                      generator=torch.Generator().manual_seed(0))


def autocast(device,amp):
    if device.type!='cuda' or amp=='none':
        return nullcontext()
    return torch.autocast('cuda',dtype=torch.bfloat16 if amp=='bf16' else torch.float16)


def rng_state(device):
    ns=np.random.get_state()
    return {'torch':torch.get_rng_state(),'python':random.getstate(),
            'numpy':[ns[0],ns[1].tolist(),int(ns[2]),int(ns[3]),float(ns[4])],
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(state,device):
    torch.set_rng_state(state['torch']);random.setstate(state['python'])
    ns=state['numpy'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    if device.type=='cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'],device)


def atomic_save(payload,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def dataset_signature(meta):
    return {k:meta[k] for k in ['schema_version','index_sha256','archive_sha256','config','fp_bits','node_dim','edge_dim']}


@torch.no_grad()
def validate(model,data_loader,device,amp,mask_ratio,mask_seed,loss_cfg):
    model.eval();total={};count=0
    for batch in data_loader:
        visibility(batch,mask_ratio,mask_seed);batch=to_device(batch,device)
        with autocast(device,amp):
            result=model(batch)
        vals=losses(batch,result,**loss_cfg)
        for name in ['posterior','prior']:
            prediction=result[name]['spectrum'];target=batch['target']
            vals[name+'_cosine']=cosine_per_graph(target,prediction,batch['bin_batch'],batch['num_graphs'])
            # Fragment-only diagnostic removes mass windows containing the precursor.
            root_bins=batch['node_bin'][batch['roots']]
            fragment=torch.ones_like(target).index_fill(0,root_bins,0.).index_fill(0,batch['others'],0.)
            vals[name+'_fragment_cosine']=cosine_per_graph(target*fragment,prediction*fragment,
                                                           batch['bin_batch'],batch['num_graphs'])
            vals[name+'_mass_error']=(segment_sum(prediction,batch['bin_batch'],batch['num_graphs'])-1).abs()
        vals['other_target']=batch['target'][batch['others']]
        vals['fragment_target_mass']=segment_sum(batch['target']*fragment,batch['bin_batch'],batch['num_graphs'])
        vals['no_fragment_target_fraction']=(vals['fragment_target_mass']<=1e-12).float()
        for k,v in vals.items():
            total[k]=total.get(k,0.)+float(v.double().sum())
        count+=batch['num_graphs']
    keys=['loss','reconstruction','prior_reconstruction','posterior_kl','expected_path_length',
          'posterior_cosine','posterior_fragment_cosine','posterior_mass_error',
          'prior_cosine','prior_fragment_cosine','prior_mass_error','other_target','fragment_target_mass','no_fragment_target_fraction']
    tensor=torch.tensor([total.get(k,0.) for k in keys]+[count],device=device,dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    if tensor[-1]<=0:
        raise ValueError('Empty evaluation split')
    return {k:float(tensor[i]/tensor[-1]) for i,k in enumerate(keys)}|{'samples':int(tensor[-1])}


def train(config_path,device_name='auto',backend=None,resume=None,initialize=None,epochs_override=None,
          data_override=None,output_override=None,workers_override=None,limit=None):
    config=json.loads(Path(config_path).read_text())
    with native_diagnostics(output_override or config['output']) as recorder:
        return _train(config_path,device_name,backend,resume,initialize,epochs_override,
                      data_override,output_override,workers_override,limit,recorder)


def _train(config_path,device_name,backend,resume,initialize,epochs_override,
           data_override,output_override,workers_override,limit,recorder):
    config=json.loads(Path(config_path).read_text())
    if epochs_override is not None: config['training']['epochs']=epochs_override
    if data_override: config['data']=str(data_override)
    if output_override: config['output']=str(output_override)
    if workers_override is not None: config['training']['workers']=workers_override
    device,rank,world=environment(device_name,backend)
    recorder.mark('distributed_ready',device,selected_device=str(device),world_size=world)
    tc=config['training'];seed=config.get('seed',20261016)
    if not 0<=tc['mask_ratio']<1 or tc.get('amp','bf16') not in ['none','bf16','fp16']:
        raise ValueError('Require 0<=mask_ratio<1 and amp in none/bf16/fp16')
    torch.set_num_threads(tc.get('cpu_threads',2))
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed+rank)
        configure_cuda(tc)
        if tc.get('amp','bf16')=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported; set amp=fp16 or none')
    output=Path(config['output'])
    if rank==0:
        if (output/'last.pt').exists() and not resume:
            raise FileExistsError('Existing training checkpoint; use --resume or a new output directory')
        output.mkdir(parents=True,exist_ok=True)
    if world>1:dist.barrier()
    train_data=SpectrumDAGDataset(config['data'],config['sources'],'train',limit)
    val_data=SpectrumDAGDataset(config['data'],config['sources'],'val',limit)
    if len(train_data)==0 or len(val_data)==0:
        raise ValueError('Empty train/validation split. Prepare more queries or check sources.')
    # Exact sharding for validation: no DistributedSampler padding duplicates.
    val_shard=Subset(val_data,list(range(rank,len(val_data),world)))
    sampler=DistributedSampler(train_data,world,rank,shuffle=True,seed=seed)
    train_loader=loader(train_data,tc['batch_size_per_gpu'],tc['workers'],device,sampler)
    val_loader=loader(val_shard,tc.get('eval_batch_size_per_gpu',tc['batch_size_per_gpu']),tc['workers'],device)
    mc=ModelConfig(**config['model']);model=SpectrumDAGModel(mc).to(device)
    if initialize and resume:
        raise ValueError('Use either --initialize or --resume')
    if initialize:
        ckpt=torch.load(initialize,map_location='cpu',weights_only=True)
        if normalized_model_config(ckpt['model_config'])!=asdict(mc):raise ValueError('Initialization architecture mismatch')
        if ckpt['dataset_signature']!=dataset_signature(train_data.meta):
            raise ValueError('Pretrain/fine-tune must use the same prepared split/data signature')
        model.load_state_dict(ckpt['model'])
    if tc.get('freeze_graph_prior',False):
        if not (initialize or resume):raise ValueError('Frozen prior requires --initialize or --resume')
        if not mc.use_spectrum:raise ValueError('Cannot train with both a frozen graph branch and use_spectrum=false')
        model.freeze_graph_prior()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=tc['learning_rate'],
                                weight_decay=tc.get('weight_decay',.0001),**optimizer_options(tc))
    accumulation=tc.get('accumulation',1)
    if accumulation<1:raise ValueError('accumulation must be positive')
    updates_per_epoch=math.ceil(len(train_loader)/accumulation)
    total_steps=tc['epochs']*updates_per_epoch;warmup=max(1,round(total_steps*tc.get('warmup_fraction',.05)))
    def schedule(step):
        if step<warmup:return (step+1)/warmup
        progress=(step-warmup)/max(1,total_steps-warmup)
        return .05+.95*.5*(1+math.cos(math.pi*min(progress,1.)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and tc.get('amp')=='fp16')
    start=0;best=float('inf');bad_epochs=0;global_step=0
    subset_signature=hashlib.sha256(json.dumps([train_data.index,val_data.index],sort_keys=True).encode()).hexdigest()
    if resume:
        ckpt=torch.load(resume,map_location='cpu',weights_only=True)
        if normalized_model_config(ckpt['model_config'])!=asdict(mc) or ckpt['dataset_signature']!=dataset_signature(train_data.meta):
            raise ValueError('Checkpoint model/data signature mismatch')
        if ckpt['world_size']!=world:
            raise ValueError('Exact resume requires the original world size; use --initialize to change it')
        if ckpt['subset_signature']!=subset_signature:
            raise ValueError('Resume sample subset/source mismatch')
        saved=ckpt['training_config'].copy();current=tc.copy()
        # IO/logging may change. Epoch count defines the cosine schedule and may not.
        for key in ['workers','log_every','cpu_threads','eval_batch_size_per_gpu','diagnostic_steps']:
            saved.pop(key,None);current.pop(key,None)
        if saved!=current:raise ValueError('Resume training settings differ; use original config')
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler']);scaler.load_state_dict(ckpt['scaler'])
        start=ckpt['epoch']+1;best=ckpt['best'];bad_epochs=ckpt['bad_epochs'];global_step=ckpt['global_step']
        restore_rng(ckpt['rng'][rank],device)
    # Same model initialization across ranks, independent stochastic dropout thereafter.
    if not resume:torch.manual_seed(seed+rank)
    ddp=DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None,
                                broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
    loss_cfg=config.get('loss',{})
    if rank==0:
        (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        print(json.dumps({'device':str(device),'world_size':world,'parameters':sum(p.numel() for p in model.parameters()),
              'architecture':mc.architecture,'heads':mc.heads,
              'attention':'chunked_fp32' if mc.architecture=='multiscale_v2' else 'torch_mha',
              'amp':tc.get('amp','bf16'),'checkpoint_attention':mc.checkpoint_attention,
              'optimizer':optimizer_options(tc),'allow_tf32':tc.get('allow_tf32',True),
              'train_samples':len(train_data),'val_samples':len(val_data),
              'global_batch':tc['batch_size_per_gpu']*world*accumulation,'torch':str(torch.__version__)}),flush=True)
    recorder.mark('training_ready',device)
    diagnostic_steps=tc.get('diagnostic_steps',2)
    if diagnostic_steps<0:raise ValueError('diagnostic_steps must be nonnegative')
    for epoch in range(start,tc['epochs']):
        ddp.train()
        if sampler:sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True);train_sum=0.;train_count=0;started=time.monotonic()
        for step,batch in enumerate(train_loader):
            diagnose=(epoch==start and diagnostic_steps>0 and
                      (step<diagnostic_steps or step+1==min(accumulation,len(train_loader))))
            mask_seed=seed+epoch*100003
            if diagnose:recorder.batch(batch,epoch,step,mask_seed)
            def milestone(stage):
                if diagnose:recorder.mark(stage,device,epoch=epoch,batch=step+1)
            visibility(batch,tc['mask_ratio'],mask_seed)
            batch=to_device(batch,device)
            milestone('batch_on_device')
            boundary=(step+1)%accumulation==0 or step+1==len(train_loader)
            group_start=(step//accumulation)*accumulation*tc['batch_size_per_gpu']
            group_samples=min(accumulation*tc['batch_size_per_gpu'],len(sampler)-group_start)
            sync=nullcontext() if boundary or world==1 else ddp.no_sync()
            with sync:
                milestone('forward_start')
                with autocast(device,tc.get('amp','bf16')):
                    predictions=ddp(batch)
                milestone('forward_done')
                values=losses(batch,predictions,**loss_cfg)
                loss=values['loss'].sum()/group_samples
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, step {step}')
                milestone('loss_done')
                milestone('backward_start')
                scaler.scale(loss).backward()
                milestone('backward_done')
            if boundary:
                milestone('optimizer_start')
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get('gradient_clip',1.),
                                                    error_if_nonfinite=True,foreach=tc.get('gradient_foreach'))
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
                milestone('optimizer_done')
            train_sum+=float(values['loss'].detach().double().sum());train_count+=batch['num_graphs']
            if rank==0 and (step+1)%tc.get('log_every',50)==0:
                print(json.dumps({'epoch':epoch,'batch':step+1,'updates':global_step,
                                  'loss':train_sum/train_count,'lr':optimizer.param_groups[0]['lr']}),flush=True)
        sums=torch.tensor([train_sum,train_count],dtype=torch.float64,device=device)
        if world>1:dist.all_reduce(sums)
        # Evaluate the underlying module, avoiding unequal-length DDP forward collectives.
        metrics=validate(model,val_loader,device,tc.get('amp','bf16'),tc['mask_ratio'],seed+7000001,loss_cfg)
        improved=metrics['reconstruction']<best-tc.get('min_delta',.00001)
        if improved:best=metrics['reconstruction'];bad_epochs=0
        else:bad_epochs+=1
        local_rng=rng_state(device)
        states=[None]*world
        if world>1:dist.all_gather_object(states,local_rng)
        else:states=[local_rng]
        if rank==0:
            log={'epoch':epoch,'global_step':global_step,'seconds':time.monotonic()-started,
                 'train_loss':float(sums[0]/sums[1]),'validation':metrics,'best':best,'bad_epochs':bad_epochs}
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(json.dumps(log),flush=True)
            payload={'model':model.state_dict(),'model_config':asdict(mc),'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'global_step':global_step,
                     'best':best,'bad_epochs':bad_epochs,'rng':states,'world_size':world,'training_config':tc,
                     'config':config,'dataset_signature':dataset_signature(train_data.meta),'torch_version':str(torch.__version__),
                     'subset_signature':subset_signature}
            atomic_save(payload,output/'last.pt')
            if improved:atomic_save(payload,output/'best.pt')
            if tc.get('checkpoint_every',0) and (epoch+1)%tc['checkpoint_every']==0:
                atomic_save(payload,output/f'epoch_{epoch:04d}.pt')
        if world>1:dist.barrier()
        if bad_epochs>=tc.get('patience',20):break
    if dist.is_initialized():dist.destroy_process_group()


def evaluate(checkpoint,data,split='test',sources=None,device_name='auto',backend=None,output=None,workers=0,limit=None):
    device,rank,world=environment(device_name,backend)
    torch.set_num_threads(2)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=ckpt['config'];tc=cfg['training']
    if device.type=='cuda':configure_cuda(tc)
    dataset=SpectrumDAGDataset(data,sources or cfg['sources'],split,limit)
    if dataset_signature(dataset.meta)!=ckpt['dataset_signature']:raise ValueError('Evaluation dataset signature mismatch')
    model=SpectrumDAGModel(ModelConfig(**ckpt['model_config'])).to(device);model.load_state_dict(ckpt['model'])
    shard=Subset(dataset,list(range(rank,len(dataset),world)))
    dl=loader(shard,tc.get('eval_batch_size_per_gpu',4),workers,device)
    result={}
    for mode,ratio in [('masked',tc['mask_ratio']),('full_context',0.)]:
        result[mode]=validate(model,dl,device,tc.get('amp','bf16'),ratio,cfg.get('seed',20261016)+7000001,cfg.get('loss',{}))
    if rank==0:
        text=json.dumps({'split':split,'sources':sources or cfg['sources'],'metrics':result},indent=2)
        print(text,flush=True)
        if output:
            Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(text+'\n')
    if dist.is_initialized():dist.destroy_process_group()
