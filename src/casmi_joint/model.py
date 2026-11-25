"""Student inference consumes peaks/conditions and a fixed catalog, never a query DAG."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from casmi_peaks.chem import ATOM_DIM,BOND_DIM,NUMERIC_DIM


@dataclass(frozen=True)
class JointConfig:
    hidden: int = 256
    heads: int = 8
    spectrum_layers: int = 4
    graph_layers: int = 3
    dropout: float = .1
    temperature: float = .07
    mass_frequencies: int = 24
    mass_tolerance: float = .007
    relation_rank: int = 48
    use_pair_bias: bool = True
    peak_only: bool = False

    def __post_init__(self):
        for name in ('hidden','heads','spectrum_layers','graph_layers','relation_rank'):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.mass_frequencies,int) or isinstance(self.mass_frequencies,bool) or self.mass_frequencies<2:
            raise ValueError('mass_frequencies must be an integer >= 2')
        if self.hidden%self.heads: raise ValueError('hidden must be divisible by heads')
        if not math.isfinite(self.dropout) or not 0<=self.dropout<1: raise ValueError('Invalid dropout')
        if not math.isfinite(self.temperature) or self.temperature<=0: raise ValueError('Invalid temperature')
        if not math.isfinite(self.mass_tolerance) or self.mass_tolerance<=0: raise ValueError('Invalid mass tolerance')
        if not isinstance(self.use_pair_bias,bool) or not isinstance(self.peak_only,bool):
            raise ValueError('Ablation flags must be boolean')


class MassFeatures(nn.Module):
    """Masses in Da, with periods spanning broad shape through measurement tolerance."""
    def __init__(self, count, tolerance):
        super().__init__()
        if not isinstance(count,int) or isinstance(count,bool) or count<2 or not math.isfinite(tolerance) or tolerance<=0:
            raise ValueError('Invalid mass encoding')
        self.register_buffer('frequency', torch.logspace(math.log10(1/2000), math.log10(1/(4*tolerance)), count))

    def forward(self, mass):
        angle = mass.float().unsqueeze(-1) * self.frequency * (2*math.pi)
        return torch.cat([angle.sin(), angle.cos()], -1)


class PeakBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__(); h = cfg.hidden
        self.heads = cfg.heads; self.width = h // cfg.heads; self.dropout = cfg.dropout
        self.norm1 = nn.LayerNorm(h); self.norm2 = nn.LayerNorm(h)
        self.qkv = nn.Linear(h, 3*h); self.output = nn.Linear(h, h)
        self.ff = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4*h, h))

    def forward(self, x, valid, bias, peak_only=False):
        b,n,h = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,self.width).permute(2,0,3,1,4).unbind(0)
        logits = (q.float() @ k.float().transpose(-1,-2))/math.sqrt(self.width) + bias.float()
        allowed=valid[:,None,None,:]
        if peak_only:
            diagonal=torch.eye(n,device=x.device,dtype=torch.bool)
            # Valid peaks read only themselves. Padding queries may read real
            # keys, so no row becomes all -inf before its output is masked.
            per_query=diagonal[None]|(~valid)[:,:,None]
            allowed=allowed&per_query[:,None]
        logits = logits.masked_fill(~allowed, -torch.inf)
        weights = F.dropout(logits.softmax(-1), self.dropout, self.training)
        value = (weights.to(v.dtype) @ v).transpose(1,2).reshape(b,n,h)
        x = x + F.dropout(self.output(value), self.dropout, self.training)
        x = x + F.dropout(self.ff(self.norm2(x)), self.dropout, self.training)
        return x.masked_fill(~valid[:,:,None], 0.)


class FragmentEncoder(nn.Module):
    """Edge-aware sum aggregation and multi-layer readout; graph/ion kept separate."""
    def __init__(self,cfg):
        super().__init__(); h=cfg.hidden
        self.atom=nn.Linear(ATOM_DIM,h); self.bond=nn.Linear(BOND_DIM,h)
        self.epsilon=nn.Parameter(torch.zeros(cfg.graph_layers))
        self.update=nn.ModuleList(nn.Sequential(nn.Linear(h,2*h),nn.GELU(),nn.Linear(2*h,h)) for _ in range(cfg.graph_layers))
        self.norm=nn.ModuleList(nn.LayerNorm(h) for _ in range(cfg.graph_layers))
        self.readout=nn.ModuleList(nn.Linear(2*h,h) for _ in range(cfg.graph_layers))
        self.ion=nn.Sequential(nn.Linear(NUMERIC_DIM,h),nn.GELU(),nn.LayerNorm(h))
        self.output=nn.Sequential(nn.Linear(2*h,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h))
        self.dropout=cfg.dropout

    def forward(self,g):
        h=self.atom(g['atoms']); source,target=g['edges']; bond=self.bond(g['bonds'])
        count=h.new_zeros(g['size']).index_add(0,g['groups'],h.new_ones(len(h))).clamp_min(1)[:,None]
        pooled=[]
        for k,(update,norm,readout) in enumerate(zip(self.update,self.norm,self.readout)):
            message=F.relu(h[source]+bond)
            aggregate=h.new_zeros(h.shape).index_add(0,target,message)
            h=norm(h+F.dropout(update((1+self.epsilon[k])*h+aggregate),self.dropout,self.training))
            total=h.new_zeros(g['size'],h.shape[-1]).index_add(0,g['groups'],h)
            pooled.append(readout(torch.cat([total/count,total/count.sqrt()],-1)))
        graph=torch.stack(pooled).mean(0)
        return F.normalize(self.output(torch.cat([graph,self.ion(g['numeric'])],-1)).float(),dim=-1)


class RelationNetwork(nn.Module):
    """Four weak teacher relations: direct, reachable, siblings, common precursor.

    Endpoint projections are factorized, avoiding a hidden-sized tensor for
    every candidate-pair. Relation scores remain teacher agreement estimates.
    """
    def __init__(self, cfg):
        super().__init__(); self.rank = cfg.relation_rank
        self.left = nn.Linear(cfg.hidden, 4*self.rank)
        self.right = nn.Linear(cfg.hidden, 4*self.rank)
        self.context = nn.Sequential(nn.Linear(8,48), nn.GELU(), nn.Linear(48,4))

    def project(self, embedding):
        return (self.left(embedding).reshape(*embedding.shape[:-1],4,self.rank),
                self.right(embedding).reshape(*embedding.shape[:-1],4,self.rank))

    def logits(self, left, right, ma, mb, condition):
        value = torch.einsum('...hr,...hr->...h',left,right)/math.sqrt(self.rank)
        return value+self.context_features(ma,mb,condition)

    def context_features(self,ma,mb,condition):
        precursor = (condition[...,1]*1000).clamp_min(1)
        delta = ma-mb
        features = torch.stack([ma/1000, mb/1000, delta/1000, delta.abs()/1000,
                                delta/precursor, condition[...,0], condition[...,2], ma/precursor], -1)
        return self.context(features.float())

    def pair_logits(self,left,right,ma,mb,condition):
        """All endpoint pairs, without materializing [N,K,S,heads,rank]."""
        value=torch.einsum('nkhr,nshr->nksh',left,right)/math.sqrt(self.rank)
        n,k,s=value.shape[:3]
        ma=ma[:,:,None].expand(n,k,s); mb=mb[:,None,:].expand(n,k,s)
        c=condition[:,None,None,:].expand(n,k,s,-1)
        return value+self.context_features(ma,mb,c)

    @staticmethod
    def compatibility(logits, reverse_logits):
        # Center the score. Neither missing teacher edges nor these scores are
        # assertions of chemical impossibility. Direction is retained in heads.
        both = .5*(logits.sigmoid()+reverse_logits.sigmoid())
        return (both*both.new_tensor([.25,.25,.15,.35])).sum(-1)*2-1


def neighbor_indices(masses, valid, count):
    """Sparse support from local mass neighbors and full-range mass anchors.

    All peaks are encoded globally before this step. Ranking peaks are sorted
    by mass upstream; no topology or teacher target enters this selection.
    """
    n = len(masses)
    if n == 0 or count <= 0: return torch.empty((n,0), dtype=torch.long, device=masses.device)
    device=masses.device
    # This selection is intentionally nondifferentiable. Copy once rather
    # than synchronizing a CUDA scalar for every neighbor comparison.
    masses=masses.detach().cpu(); valid=valid.detach().cpu()
    count = min(count, int(valid.sum()))
    if count == 0: return torch.empty((n,0), dtype=torch.long, device=device)
    distance = (masses[:,None]-masses[None,:]).abs()
    distance = distance.masked_fill(~valid[None,:], torch.inf)
    distance.fill_diagonal_(torch.inf)
    # Local mass support plus spread-out global anchors; selection is fixed
    # by observed masses and validity, not by pseudo-label coverage.
    local_n = (count+1)//2
    rows=[]
    available = [i for i in masses.argsort(stable=True).tolist() if bool(valid[i])]
    for i in range(n):
        nearest=[j for j in distance[i].argsort(stable=True).tolist() if j!=i and bool(valid[j])]
        choices=nearest[:local_n]
        remaining=[j for j in available if j!=i and j not in choices]
        slots=min(count-len(choices),len(remaining))
        if slots==1:
            choices.append(max(remaining,key=lambda j:float(distance[i,j])))
        elif slots>1:
            locations=torch.linspace(0,len(remaining)-1,slots).round().long().tolist()
            choices.extend(remaining[p] for p in locations)
        for j in nearest:
            if len(choices)>=count: break
            if j not in choices: choices.append(j)
        while len(choices)<count: choices.append(i)  # masked by support validity
        rows.append(choices[:count])
    return torch.tensor(rows,device=device,dtype=torch.long)


def joint_update(unary, embeddings, masses, valid, peak_masses, conditions, relation,
                 strength=.5, iterations=2, neighbors=6, support_k=16, damping=.5,support_valid=None):
    """Mean-field-like damped updates on one observed spectrum's candidate lists.

    No teacher labels, DAG edges or candidate injection are accepted here.
    Scores stay in logit units. With strength=0 this is exactly unary scoring.
    """
    if not math.isfinite(strength) or strength<0 or not math.isfinite(damping) or not 0<damping<=1:
        raise ValueError('Invalid joint inference settings')
    for name,value,minimum in (('iterations',iterations,0),('neighbors',neighbors,1),('support_k',support_k,1)):
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum: raise ValueError(f'Invalid {name}')
    if unary.ndim!=2 or valid.shape!=unary.shape or valid.dtype!=torch.bool: raise ValueError('Invalid candidate mask')
    if support_valid is not None and (support_valid.shape!=valid.shape or support_valid.dtype!=torch.bool):
        raise ValueError('Invalid support candidate mask')
    if strength == 0 or iterations == 0 or len(unary)<2: return unary
    n,k=unary.shape; active=valid.any(-1)
    sender_valid=valid if support_valid is None else valid&support_valid
    sender_active=sender_valid.any(-1)
    safe=unary.masked_fill(~valid,-torch.inf)
    # An empty row has no label and sends no message; a finite dummy avoids NaN.
    safe=torch.where(active[:,None],safe,torch.zeros_like(safe))
    current=safe; left,right=relation.project(embeddings)
    adjacent=neighbor_indices(peak_masses,sender_active,neighbors)
    if adjacent.shape[1]==0: return unary
    for _ in range(iterations):
        sender_logits=current.masked_fill(~sender_valid,-torch.inf)
        sender_logits=torch.where(sender_active[:,None],sender_logits,torch.zeros_like(sender_logits))
        probability=sender_logits.softmax(-1)*sender_valid
        messages=torch.zeros_like(unary); denominator=unary.new_zeros(n)
        for slot in range(adjacent.shape[1]):
            j=adjacent[:,slot]
            values,which=probability[j].topk(min(support_k,k),dim=-1)
            selected_valid=sender_valid[j].gather(1,which)
            values=values*selected_valid
            values=values/values.sum(-1,keepdim=True).clamp_min(1e-12)
            rb=right[j].gather(1,which[:,:,None,None].expand(-1,-1,4,relation.rank))
            lb=left[j].gather(1,which[:,:,None,None].expand(-1,-1,4,relation.rank))
            mb=masses[j].gather(1,which)
            if hasattr(relation,'pair_logits'):
                logits=relation.pair_logits(left,rb,masses,mb,conditions)
                reverse=relation.pair_logits(lb,right,mb,masses,conditions).transpose(1,2)
            else:  # Lightweight relation doubles used in contract tests.
                ma=masses[:,:,None]; mb=mb[:,None,:]
                c=conditions[:,None,None,:].expand(-1,k,which.shape[1],-1)
                logits=relation.logits(left[:,:,None],rb[:,None],ma.expand_as(c[...,0]),mb.expand_as(c[...,0]),c)
                reverse=relation.logits(lb[:,None],right[:,:,None],mb.expand_as(c[...,0]),ma.expand_as(c[...,0]),c)
            compat=relation.compatibility(logits,reverse)
            support=sender_active[j] & (j!=torch.arange(n,device=j.device))
            # Entropy-dependent reliability, bounded away from absolute certainty.
            entropy=-(values*values.clamp_min(1e-12).log()).sum(-1)
            number=selected_valid.sum(-1)
            relative_entropy=entropy/number.clamp_min(2).to(entropy.dtype).log()
            reliability=torch.where(number>1,(1-relative_entropy).clamp(.1,1.),torch.ones_like(entropy))
            reliability=reliability*(number>0)*support
            messages=messages+(compat*values[:,None,:]).sum(-1)*reliability[:,None]
            denominator=denominator+reliability
        messages=messages/denominator[:,None].clamp_min(1.)
        # Remove candidate-independent offsets before mixing with unary evidence.
        center=(messages*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        proposal=(safe+strength*(messages-center)).masked_fill(~valid,-torch.inf)
        proposal=torch.where(active[:,None],proposal,torch.zeros_like(proposal))
        current=proposal if damping==1 else (1-damping)*current+damping*proposal
    return current.masked_fill(~valid,-torch.inf)


class JointAnnotator(nn.Module):
    def __init__(self,cfg=JointConfig()):
        super().__init__(); self.cfg=cfg; h=cfg.hidden
        if h%cfg.heads or min(cfg.spectrum_layers,cfg.graph_layers,cfg.relation_rank)<1 or cfg.temperature<=0:
            raise ValueError('Invalid joint model configuration')
        self.mass=MassFeatures(cfg.mass_frequencies,cfg.mass_tolerance)
        self.token=nn.Sequential(nn.Linear(5+4*cfg.mass_frequencies,h),nn.GELU(),nn.LayerNorm(h))
        self.condition=nn.Sequential(nn.Linear(3,h),nn.GELU(),nn.Linear(h,h))
        self.pair=nn.Sequential(nn.Linear(2*cfg.mass_frequencies+2,32),nn.GELU(),nn.Linear(32,cfg.heads))
        self.layers=nn.ModuleList(PeakBlock(cfg) for _ in range(cfg.spectrum_layers))
        self.query=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Linear(h,h))
        self.fragment=FragmentEncoder(cfg); self.relation=RelationNetwork(cfg)

    def encode_spectrum(self,batch):
        peaks=batch['peaks']; valid=batch['valid']; c=batch['conditions']
        if peaks.ndim!=3 or peaks.shape[-1]!=2 or valid.shape!=peaks.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('Invalid spectrum tensor/mask shape')
        if peaks.shape[0]==0 or peaks.shape[1]==0 or not bool(valid.any(1).all()): raise ValueError('Empty spectrum')
        if c.shape!=(peaks.shape[0],3) or not bool(torch.isfinite(c).all()) or not bool((c[:,1]>0).all()):
            raise ValueError('Invalid spectrum conditions')
        observed=peaks[valid]
        if not bool(torch.isfinite(observed).all()) or not bool((observed[:,0]>0).all()) or not bool((observed[:,1]>=0).all()):
            raise ValueError('Invalid observed peaks')
        peaks=torch.where(valid[:,:,None],peaks,torch.zeros_like(peaks))
        mz=peaks[...,0]; intensity=peaks[...,1]
        rel=intensity/intensity.amax(1,keepdim=True).clamp_min(1e-12)
        precursor=(c[:,1:2]*1000).clamp_min(1)
        base=torch.stack([mz/1000,mz/precursor,(precursor-mz)/1000,rel.sqrt(),torch.log1p(9*rel)/math.log(10)],-1)
        x=self.token(torch.cat([base,self.mass(mz),self.mass(precursor-mz)],-1))+self.condition(c)[:,None]
        delta=mz[:,:,None]-mz[:,None,:]
        pair=torch.cat([self.mass(delta),delta[:,:,:,None]/1000,delta.abs()[:,:,:,None]/1000],-1)
        bias=self.pair(pair).permute(0,3,1,2)
        if not self.cfg.use_pair_bias: bias=bias*0.
        for layer in self.layers: x=layer(x,batch['valid'],bias,self.cfg.peak_only)
        return F.normalize(self.query(x).float(),dim=-1)

    def forward(self,batch,graphs,targets,relations,joint):
        queries=self.encode_spectrum(batch); fragments=self.fragment(graphs)
        pos=targets['positions']; selected=queries[pos[:,0],pos[:,1]]
        candidate=fragments[targets['indices']]
        unary=(selected[:,None]*candidate).sum(-1)/self.cfg.temperature
        unary=unary.masked_fill(~targets['valid'],-torch.inf)
        scores=unary
        if joint.get('strength',0):
            pieces=[]
            for b in range(len(batch['peaks'])):
                rows=torch.where(pos[:,0]==b)[0]
                pieces.append((rows,joint_update(unary[rows],candidate[rows],targets['masses'][rows],targets['valid'][rows],
                    batch['peaks'][b,pos[rows,1],0],batch['conditions'][b].expand(len(rows),-1),self.relation,
                    support_valid=targets.get('support_valid',targets['valid'])[rows],**joint)))
            scores=torch.zeros_like(unary)
            for rows,value in pieces: scores=scores.index_copy(0,rows,value)
        def objective(value):
            supervised=targets['supervised']
            value=value[supervised]; positive=targets['positive'][supervised]
            norm=torch.logsumexp(value,dim=-1)
            if targets['mode']=='set':
                return (norm-torch.logsumexp(value.masked_fill(~positive,-torch.inf),dim=-1)).mean()
            logprob=(value-norm[:,None]).masked_fill(~positive,0.)
            return -(targets['weights'][supervised]*logprob).sum(-1).mean()
        left,right=self.relation.project(fragments)
        r=relations
        logits=self.relation.logits(left[r['left']],right[r['right']],r['ma'],r['mb'],r['conditions'])
        raw=F.binary_cross_entropy_with_logits(logits,r['labels'],reduction='none')
        head_weights=r.get('head_weights',torch.ones_like(raw))
        if head_weights.shape!=raw.shape or not bool(torch.isfinite(head_weights).all()) or bool((head_weights<0).any()):
            raise ValueError('Invalid relation head weights')
        effective=r['weights'][:,None]*head_weights
        relation_loss=(raw*effective).sum()/effective.sum().clamp_min(1e-12)
        return dict(peak_loss=objective(scores),unary_loss=objective(unary),relation_loss=relation_loss)
