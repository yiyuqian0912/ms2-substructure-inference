"""The query encoder sees only peaks and acquisition conditions, never a DAG."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .chem import ATOM_DIM,BOND_DIM,NUMERIC_DIM


@dataclass(frozen=True)
class ModelConfig:
    hidden:int=256
    heads:int=8
    spectrum_layers:int=4
    graph_layers:int=3
    fourier_frequencies:int=12
    dropout:float=.1
    temperature:float=.07


class Fourier(nn.Module):
    def __init__(self,n):
        super().__init__();self.register_buffer('frequency',2.**torch.arange(n)*math.pi)

    def forward(self,x):
        angle=x.unsqueeze(-1)*self.frequency
        return torch.cat([angle.sin(),angle.cos()],-1)


class FragmentEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();h=cfg.hidden
        self.atom=nn.Linear(ATOM_DIM,h);self.bond=nn.Linear(BOND_DIM,h)
        self.message=nn.ModuleList(nn.Sequential(nn.Linear(2*h,h),nn.GELU(),nn.Linear(h,h)) for _ in range(cfg.graph_layers))
        self.update=nn.ModuleList(nn.Sequential(nn.Linear(2*h,h),nn.GELU(),nn.Dropout(cfg.dropout)) for _ in range(cfg.graph_layers))
        self.norm=nn.ModuleList(nn.LayerNorm(h) for _ in range(cfg.graph_layers))
        self.readout=nn.Sequential(nn.Linear(2*h+NUMERIC_DIM,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h))

    def forward(self,g):
        h=self.atom(g['atoms']);source,target=g['edges'];bond=self.bond(g['bonds'])
        degree=h.new_zeros(len(h)).index_add(0,target,h.new_ones(len(target))).clamp_min(1)
        for msg,update,norm in zip(self.message,self.update,self.norm):
            messages=msg(torch.cat([h[source],bond],-1))
            aggregate=h.new_zeros(h.shape).index_add(0,target,messages)/degree[:,None]
            h=norm(h+update(torch.cat([h,aggregate],-1)))
        total=h.new_zeros(g['size'],h.shape[-1]).index_add(0,g['groups'],h)
        count=h.new_zeros(g['size']).index_add(0,g['groups'],h.new_ones(len(h)))
        pooled=torch.cat([total/count[:,None].clamp_min(1),total/torch.sqrt(count[:,None].clamp_min(1)),g['numeric']],-1)
        return F.normalize(self.readout(pooled).float(),dim=-1)


class PeakAnnotator(nn.Module):
    def __init__(self,cfg=ModelConfig()):
        super().__init__()
        if cfg.hidden%cfg.heads or cfg.spectrum_layers<1 or cfg.graph_layers<1 or cfg.temperature<=0:
            raise ValueError('Invalid model configuration')
        self.cfg=cfg;h=cfg.hidden;self.fourier=Fourier(cfg.fourier_frequencies)
        self.token=nn.Sequential(nn.Linear(5+4*cfg.fourier_frequencies,h),nn.GELU(),nn.LayerNorm(h))
        self.condition=nn.Sequential(nn.Linear(3+4*cfg.fourier_frequencies,h),nn.GELU(),nn.Linear(h,h))
        layer=nn.TransformerEncoderLayer(h,cfg.heads,4*h,cfg.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.spectrum=nn.TransformerEncoder(layer,cfg.spectrum_layers,enable_nested_tensor=False)
        self.query=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Linear(h,h))
        self.fragment=FragmentEncoder(cfg)

    def encode_spectrum(self,batch):
        peaks=batch['peaks'];condition=batch['conditions'];mz=peaks[:,:,0];intensity=peaks[:,:,1]
        relative=intensity/intensity.amax(1,keepdim=True).clamp_min(1e-12)
        precursor=condition[:,1:2]*1000.
        base=torch.stack([mz/1000.,mz/precursor.clamp_min(1),
                          (precursor-mz)/1000.,relative.sqrt(),torch.log1p(9*relative)/math.log(10)],-1)
        features=torch.cat([base,self.fourier(mz/1000.),self.fourier(mz/precursor.clamp_min(1))],-1)
        c=torch.cat([condition,self.fourier(condition[:,0]),self.fourier(condition[:,1])],-1)
        token=self.token(features)+self.condition(c)[:,None,:]
        encoded=self.spectrum(token,src_key_padding_mask=~batch['valid'])
        return F.normalize(self.query(encoded).float(),dim=-1)

    def forward(self,batch,graphs):
        return self.encode_spectrum(batch),self.fragment(graphs)


def retrieval_loss(queries,fragments,targets,temperature,mode):
    pos=targets['positions'];q=queries[pos[:,0],pos[:,1]]
    scores=(q[:,None,:]*fragments[targets['indices']]).sum(-1)/temperature
    scores=scores.masked_fill(~targets['valid'],-torch.inf)
    normalization=torch.logsumexp(scores,dim=1)
    if mode=='set':
        loss=normalization-torch.logsumexp(scores.masked_fill(~targets['positive'],-torch.inf),dim=1)
    elif mode=='soft':
        logprob=(scores-normalization[:,None]).masked_fill(~targets['positive'],0.)
        loss=-(targets['weights']*logprob).sum(1)
    else:raise ValueError('Loss mode must be soft or set')
    return loss.mean()
