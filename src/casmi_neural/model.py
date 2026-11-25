from dataclasses import dataclass
import math

import torch
from torch import nn

from .data import FP_BITS,NODE_DIM,EDGE_DIM
from .flow import action_probabilities,propagate,segment_sum


@dataclass(frozen=True)
class ModelConfig:
    hidden: int=192
    heads: int=6
    dag_layers: int=4
    spectrum_layers: int=3
    cross_layers: int=2
    dropout: float=.1
    use_fingerprints: bool=True
    use_spectrum: bool=True
    architecture: str='legacy'
    fourier_frequencies: int=16
    rope_max_frequency: float=32768.
    region_bins: tuple=(16,64)
    attention_chunk: int=128
    checkpoint_attention: bool=True
    use_rope: bool=True
    use_multiscale: bool=True
    dag_attention: bool=True
    posterior_dag_layers: int=1

    def __post_init__(self):
        object.__setattr__(self,'region_bins',tuple(self.region_bins))
        if self.architecture not in ['legacy','multiscale_v2']:
            raise ValueError('Unknown model architecture')
        if self.hidden<1 or self.heads<1 or self.hidden%self.heads:
            raise ValueError('hidden must be positive and divisible by heads')
        if not 0<=self.dropout<1 or self.dag_layers<0 or self.posterior_dag_layers<0:
            raise ValueError('Invalid dropout or DAG depth')
        if not 1<=self.fourier_frequencies<=20 or not math.isfinite(self.rope_max_frequency) or self.rope_max_frequency<1:
            raise ValueError('Invalid mass encoding frequencies')
        if self.attention_chunk<1 or not self.region_bins or any(type(n) is not int or n<1 for n in self.region_bins):
            raise ValueError('Invalid attention chunk or region grid')


def normalized_model_config(config):
    """Old checkpoints omit v2 options and always retain legacy semantics."""
    from dataclasses import asdict
    return asdict(ModelConfig(**config))


def mlp(inp,out,dropout=0.):
    return nn.Sequential(nn.Linear(inp,out),nn.GELU(),nn.Dropout(dropout),nn.Linear(out,out))


class DAGLayer(nn.Module):
    def __init__(self,hidden,dropout):
        super().__init__()
        self.down=mlp(hidden*3,hidden,dropout);self.up=mlp(hidden*3,hidden,dropout)
        self.update=mlp(hidden*3,hidden,dropout);self.norm=nn.LayerNorm(hidden)

    def forward(self,h,edge,edge_index):
        source,child=edge_index
        x=torch.cat([h[source],h[child],edge],-1)
        down=segment_sum(self.down(x),child,len(h))
        up=segment_sum(self.up(x),source,len(h))
        down=down/segment_sum(h.new_ones(len(child),1),child,len(h)).clamp_min(1)
        up=up/segment_sum(h.new_ones(len(source),1),source,len(h)).clamp_min(1)
        return self.norm(h+self.update(torch.cat([h,down,up],-1)))


class ActionHead(nn.Module):
    def __init__(self,h,dropout):
        super().__init__()
        self.stop=nn.Linear(h,1);self.edge=nn.Sequential(mlp(3*h,h,dropout),nn.Linear(h,1))
        self.other=nn.Linear(h,1)

    def forward(self,node,edge,edge_index,graph_context):
        source,child=edge_index
        return (self.stop(node).squeeze(-1),
                self.edge(torch.cat([node[source],node[child],edge],-1)).squeeze(-1),
                self.other(graph_context).squeeze(-1))


class SpectrumDAGModel(nn.Module):
    def __init__(self,cfg=ModelConfig()):
        super().__init__();self.cfg=cfg;h=cfg.hidden
        if h%cfg.heads:
            raise ValueError('hidden must be divisible by heads')
        if cfg.spectrum_layers<1 or cfg.cross_layers<1:
            raise ValueError('spectrum_layers and cross_layers must be >=1; use use_spectrum=false for ablation')
        if cfg.architecture=='multiscale_v2':
            from .model_v2 import MultiScaleSpectrumDAG
            self.v2=MultiScaleSpectrumDAG(cfg)
            return
        self.fp=nn.Linear(FP_BITS,h,bias=False);self.node_num=mlp(NODE_DIM,h)
        self.meta=mlp(3,h);self.edge_embed=mlp(EDGE_DIM,h)
        self.node_norm=nn.LayerNorm(h)
        self.dag=nn.ModuleList(DAGLayer(h,cfg.dropout) for _ in range(cfg.dag_layers))
        self.register_buffer('frequencies',torch.logspace(0,3,24),persistent=True)
        self.peak=mlp(2+48,h)
        layer=nn.TransformerEncoderLayer(h,cfg.heads,h*4,cfg.dropout,batch_first=True,norm_first=True)
        self.spectrum=nn.TransformerEncoder(layer,cfg.spectrum_layers,enable_nested_tensor=False)
        self.cls=nn.Parameter(torch.zeros(h));nn.init.normal_(self.cls,std=.02)
        self.cross=nn.ModuleList(nn.MultiheadAttention(h,cfg.heads,cfg.dropout,batch_first=True) for _ in range(cfg.cross_layers))
        self.cross_norm=nn.ModuleList(nn.LayerNorm(h) for _ in range(cfg.cross_layers))
        self.cross_ff=nn.ModuleList(mlp(h,h,cfg.dropout) for _ in range(cfg.cross_layers))
        self.prior_head=ActionHead(h,cfg.dropout);self.posterior_head=ActionHead(h,cfg.dropout)
        # Start posterior=prior. Residual heads learn observation-specific changes.
        nn.init.zeros_(self.posterior_head.stop.weight);nn.init.zeros_(self.posterior_head.stop.bias)
        nn.init.zeros_(self.posterior_head.edge[-1].weight);nn.init.zeros_(self.posterior_head.edge[-1].bias)
        nn.init.zeros_(self.posterior_head.other.weight);nn.init.zeros_(self.posterior_head.other.bias)
        self._freeze_prior=False

    def freeze_graph_prior(self):
        if hasattr(self,'v2'):
            self.v2.freeze_graph_prior();return
        self._freeze_prior=True
        for module in self.graph_prior_modules():
            module.requires_grad_(False);module.eval()

    def graph_prior_modules(self):
        if hasattr(self,'v2'):return self.v2.graph_prior_modules()
        return [self.fp,self.node_num,self.meta,self.edge_embed,self.node_norm,self.dag,self.prior_head]

    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'_freeze_prior',False):
            for module in self.graph_prior_modules():module.eval()
        return self

    def forward(self,batch):
        if hasattr(self,'v2'):return self.v2(batch)
        cfg=self.cfg;size=batch['num_graphs'];nb=batch['node_batch']
        fp=batch['node_fp'] if cfg.use_fingerprints else batch['node_fp']*0
        meta=self.meta(batch['meta'])
        node=self.node_norm(self.fp(fp)+self.node_num(batch['node_num'])+meta[nb])
        edge=self.edge_embed(batch['edge_attr'])
        for layer in self.dag:
            node=layer(node,edge,batch['edge_index'])
        context=segment_sum(node,nb,size)/segment_sum(node.new_ones(len(node),1),nb,size)
        prior_logits=self.prior_head(node,edge,batch['edge_index'],context)
        # Hidden peaks have neither mass nor intensity available to the encoder.
        visible=batch['visible']
        intensity=batch['tokens'][...,1]*visible
        intensity=intensity/intensity.sum(1,keepdim=True).clamp_min(1e-12)
        mz=batch['tokens'][...,0]*visible
        angle=mz[...,None]*self.frequencies*math.pi
        features=torch.cat([mz[...,None],torch.log1p(100*intensity)[...,None],angle.sin(),angle.cos()],-1)
        token=self.peak(features)+meta[:,None]
        token=torch.cat([self.cls.view(1,1,-1).expand(size,1,-1)+meta[:,None],token],1)
        padding=torch.cat([torch.zeros(size,1,dtype=torch.bool,device=node.device),~visible],1)
        token=self.spectrum(token,src_key_padding_mask=padding)
        # Convert disjoint graph nodes to a padded batch for cross attention.
        counts=torch.bincount(nb,minlength=size)
        # Counts/maxima are calculated on CPU in collate, not via a CUDA sync.
        maximum=max(len(r['node_num']) for r in batch['records'])
        offset=counts.cumsum(0)-counts
        local=torch.arange(len(node),device=node.device)-offset[nb]
        padded=node.new_zeros(size,maximum,cfg.hidden)
        padded=padded.index_put((nb,local),node)
        for attention,norm,ff in zip(self.cross,self.cross_norm,self.cross_ff):
            update,_=attention(padded,token,token,key_padding_mask=padding,need_weights=False)
            padded=norm(padded+update);padded=norm(padded+ff(padded))
        conditional=padded[nb,local]
        conditional_context=segment_sum(conditional,nb,size)/counts[:,None]
        residual=self.posterior_head(conditional,edge,batch['edge_index'],conditional_context)
        # Zero multiplication retains all parameters in DDP's autograd graph.
        multiplier=1. if cfg.use_spectrum else 0.
        posterior_logits=tuple(a+multiplier*b for a,b in zip(prior_logits,residual))
        output={}
        # Probability propagation and spectral losses stay in FP32 under AMP.
        with torch.autocast(device_type=node.device.type,enabled=False):
            for name,logits in [('prior',prior_logits),('posterior',posterior_logits)]:
                stop,transition=action_probabilities(logits[0],logits[1],batch['edge_index'][0])
                other=logits[2].float().sigmoid()
                output[name]=propagate(batch,stop,transition,other)
        return output
