"""Typed mass encodings, sparse directed graph attention and multi-scale memories."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .attention import MassFourier, SpectrumBlock, RoleCrossBlock, region_bank
from .data import FP_BITS, NODE_DIM, EDGE_DIM, ELEMENTS
from .flow import action_probabilities, propagate, segment_sum


class DirectedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.width, self.dropout = cfg.heads, cfg.hidden//cfg.heads, cfg.dropout
        self.query, self.key, self.value = (nn.Linear(cfg.hidden, cfg.hidden) for _ in range(3))
        self.edge_key, self.edge_value = nn.Linear(cfg.hidden, cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden)
        self.output = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, node, edge, source, target):
        n = len(node)
        q = self.query(node).reshape(n, self.heads, self.width)[target]
        k = self.key(node).reshape(n, self.heads, self.width)[source] + self.edge_key(edge).reshape(-1, self.heads, self.width)
        v = self.value(node).reshape(n, self.heads, self.width)[source] + self.edge_value(edge).reshape(-1, self.heads, self.width)
        with torch.autocast(device_type=node.device.type, enabled=False):
            logits = (q.float()*k.float()).sum(-1)/math.sqrt(self.width)
            ids = target[:, None].expand(-1, self.heads)
            maximum = logits.new_full((n, self.heads), -torch.inf).scatter_reduce(0, ids, logits, reduce='amax', include_self=True)
            weight = (logits-maximum[target].detach()).exp()
            weight = weight/segment_sum(weight, target, n)[target].clamp_min(1e-12)
            weight = F.dropout(weight, self.dropout, self.training)
            result = segment_sum(weight[..., None]*v.float(), target, n).flatten(1)
        return self.output(result.to(node.dtype))


class DirectedGraphBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.parent, self.child = DirectedAttention(cfg), DirectedAttention(cfg)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.hidden), nn.LayerNorm(cfg.hidden)
        self.combine = nn.Linear(cfg.hidden*2, cfg.hidden)
        self.ff = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden*4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden*4, cfg.hidden))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, node, edge, edge_index):
        source, target = edge_index
        value = self.norm1(node)
        update = self.combine(torch.cat((self.parent(value, edge, source, target), self.child(value, edge, target, source)), -1))
        node = node + self.dropout(update)
        return node + self.dropout(self.ff(self.norm2(node)))


class MultiScaleSpectrumDAG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Imported after model.py has defined its shared legacy building blocks.
        from .model import ActionHead, DAGLayer, mlp
        self.cfg, self._freeze_prior = cfg, False
        h, elements = cfg.hidden, len(ELEMENTS)
        self.mass_features = MassFourier(3, cfg.fourier_frequencies)
        self.edge_mass_features = MassFourier(2, cfg.fourier_frequencies)
        # NCE is a smooth experimental condition, not a fine-resolution mass axis.
        self.condition_features = MassFourier(2, min(4, cfg.fourier_frequencies))
        self.fp = nn.Sequential(nn.Linear(FP_BITS, h, bias=False), nn.LayerNorm(h))
        self.chemical = mlp(elements+3, h)
        self.topology = mlp(5, h)
        self.node_mass = mlp(self.mass_features.output_dim, h)
        self.meta = mlp(self.condition_features.output_dim+1, h)
        self.edge_embed = mlp(EDGE_DIM+self.edge_mass_features.output_dim, h)
        self.node_norm = nn.LayerNorm(h)
        self.dag = nn.ModuleList(DirectedGraphBlock(cfg) if cfg.dag_attention else DAGLayer(h, cfg.dropout) for _ in range(cfg.dag_layers))
        self.peak = mlp(self.mass_features.output_dim+2, h)
        self.spectrum = nn.ModuleList(SpectrumBlock(cfg) for _ in range(cfg.spectrum_layers))
        self.cls = nn.Parameter(torch.randn(h)*.02)
        self.region_types = nn.Parameter(torch.randn(len(cfg.region_bins), h)*.02)
        self.global_types = nn.Parameter(torch.randn(3, h)*.02)
        self.cross = nn.ModuleList(RoleCrossBlock(cfg) for _ in range(cfg.cross_layers))
        self.posterior_graph = nn.ModuleList(DirectedGraphBlock(cfg) if cfg.dag_attention else DAGLayer(h, cfg.dropout) for _ in range(cfg.posterior_dag_layers))
        self.prior_head, self.posterior_head = ActionHead(h, cfg.dropout), ActionHead(h, cfg.dropout)
        nn.init.zeros_(self.posterior_head.stop.weight); nn.init.zeros_(self.posterior_head.stop.bias)
        nn.init.zeros_(self.posterior_head.edge[-1].weight); nn.init.zeros_(self.posterior_head.edge[-1].bias)
        nn.init.zeros_(self.posterior_head.other.weight); nn.init.zeros_(self.posterior_head.other.bias)

    def graph_prior_modules(self):
        return [self.mass_features, self.edge_mass_features, self.condition_features,
                self.fp, self.chemical, self.topology, self.node_mass, self.meta,
                self.edge_embed, self.node_norm, self.dag, self.prior_head]

    def freeze_graph_prior(self):
        self._freeze_prior = True
        for module in self.graph_prior_modules():
            module.requires_grad_(False); module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_prior:
            for module in self.graph_prior_modules():
                module.eval()
        return self

    def forward(self, batch):
        cfg, size, nb = self.cfg, batch['num_graphs'], batch['node_batch']
        numeric, conditions = batch['node_num'], batch['meta']
        condition = self.meta(torch.cat((self.condition_features(conditions[:, [0, 2]]), conditions[:, 1:2]), -1))
        mass_ratio = numeric[:, len(ELEMENTS)]
        mass_absolute = mass_ratio*conditions[nb, 2]  # Units: 1000 Da, not a list position.
        mass_input = torch.stack((mass_absolute, mass_ratio, conditions[nb, 2]-mass_absolute), -1)
        chemical = torch.cat((numeric[:, :len(ELEMENTS)], numeric[:, len(ELEMENTS)+1:len(ELEMENTS)+4]), -1)
        node = self.node_norm(self.fp(batch['node_fp'] if cfg.use_fingerprints else batch['node_fp']*0)
                              + self.chemical(chemical) + self.topology(numeric[:, -5:])
                              + self.node_mass(self.mass_features(mass_input)) + condition[nb])
        source, target = batch['edge_index']
        edge_positions = torch.stack((mass_absolute[source]-mass_absolute[target], mass_ratio[source]-mass_ratio[target]), -1)
        edge = self.edge_embed(torch.cat((batch['edge_attr'], self.edge_mass_features(edge_positions)), -1))
        for layer in self.dag:
            node = layer(node, edge, batch['edge_index'])
        counts = batch['node_counts']
        context = segment_sum(node, nb, size)/counts[:, None]
        prior_logits = self.prior_head(node, edge, batch['edge_index'], context)

        # Sanitizing before all position encodings and pooling prevents masked-mass leakage.
        visible = batch['visible']
        ratio = torch.where(visible, batch['tokens'][..., 0], torch.zeros_like(batch['tokens'][..., 0]))
        intensity = torch.where(visible, batch['tokens'][..., 1], torch.zeros_like(batch['tokens'][..., 1]))
        intensity = intensity/intensity.sum(1, keepdim=True).clamp_min(1e-12)
        absolute = ratio*conditions[:, 2:3]
        neutral_loss = torch.where(visible, conditions[:, 2:3]-absolute, torch.zeros_like(absolute))
        features = torch.cat((self.mass_features(torch.stack((absolute, ratio, neutral_loss), -1)),
                              intensity[..., None], torch.log1p(100*intensity)[..., None]), -1)
        token = self.peak(features)+condition[:, None]
        token = token.masked_fill(~visible[..., None], 0.)
        token = torch.cat((self.cls[None, None].expand(size, 1, -1)+condition[:, None], token), 1)
        valid = torch.cat((torch.ones(size, 1, device=node.device, dtype=torch.bool), visible), 1)
        positions = torch.cat((absolute.new_zeros(size, 1), absolute), 1)
        for layer in self.spectrum:
            token = layer(token, positions, valid)
        regions, centers, occupied = [], [], []
        for i, bins in enumerate(cfg.region_bins):
            values, center, exists = region_bank(token[:, 1:], absolute, ratio, intensity, visible, bins)
            regions.append(values+self.region_types[i]); centers.append(center); occupied.append(exists)
        # Multiple typed summaries give the global heads a nontrivial choice;
        # attention over a single CLS key would make Q/K weights redundant.
        mean = token[:, 1:].float().sum(1)/visible.sum(1, keepdim=True).clamp_min(1)
        weighted = (token[:, 1:].float()*intensity[..., None]).sum(1)
        global_tokens = torch.stack((token[:, 0], mean.to(token.dtype), weighted.to(token.dtype)), 1)+self.global_types
        exists = visible.any(1)
        global_valid = torch.stack((torch.ones_like(exists), exists, exists), 1)
        banks = ((token[:, 1:], absolute, visible),
                 (torch.cat(regions, 1), torch.cat(centers, 1), torch.cat(occupied, 1)),
                 (global_tokens, global_valid))
        padded = node.new_zeros(size, batch['max_nodes'], cfg.hidden).index_put((nb, batch['node_local']), node)
        node_mass = mass_absolute.new_zeros(size, batch['max_nodes']).index_put((nb, batch['node_local']), mass_absolute)
        for layer in self.cross:
            padded = layer(padded, node_mass, banks)
        conditional = padded[nb, batch['node_local']]
        for layer in self.posterior_graph:
            conditional = layer(conditional, edge, batch['edge_index'])
        conditional_context = segment_sum(conditional, nb, size)/counts[:, None]
        residual = self.posterior_head(conditional, edge, batch['edge_index'], conditional_context)
        posterior = tuple(a + (1. if cfg.use_spectrum else 0.)*b for a, b in zip(prior_logits, residual))
        result = {}
        with torch.autocast(device_type=node.device.type, enabled=False):
            for name, logits in [('prior', prior_logits), ('posterior', posterior)]:
                stop, transition = action_probabilities(logits[0], logits[1], source)
                result[name] = propagate(batch, stop, transition, logits[2].float().sigmoid())
        return result
