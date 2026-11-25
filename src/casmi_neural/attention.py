"""Explicit FP32 multi-head attention, with bounded/recomputed query blocks.

This follows the stable attention strategy in the current RIND model rather
than its earlier fused SDPA implementation. It never calls fused SDPA kernels.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _attention_block(query, key, value, valid, dropout):
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = query.float() @ key.float().transpose(-2, -1) / math.sqrt(query.shape[-1])
        mask = valid[:, None, None, :]
        # Finite sentinel + explicit zeroing also handles an entirely hidden bank.
        weights = scores.masked_fill(~mask, torch.finfo(torch.float32).min).softmax(-1)
        weights = weights.masked_fill(~mask, 0.)
        weights = F.dropout(weights, p=dropout, training=dropout > 0)
        return (weights @ value.float()).to(value.dtype)


def stable_attention(query, key, value, valid, chunk_size=128, dropout=0., recompute=False):
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError('Expected [batch, heads, tokens, width] attention tensors')
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError('Query and key dimensions differ')
    if valid.shape != (key.shape[0], key.shape[2]) or valid.dtype != torch.bool:
        raise ValueError('Key validity mask must be bool [batch, keys]')
    if chunk_size < 1 or not 0 <= dropout < 1:
        raise ValueError('Invalid attention chunk/dropout')
    pieces = []
    for start in range(0, query.shape[2], chunk_size):
        q = query[:, :, start:start+chunk_size]
        if recompute and torch.is_grad_enabled() and any(t.requires_grad for t in (q, key, value)):
            piece = checkpoint(_attention_block, q, key, value, valid, dropout,
                               use_reentrant=False, preserve_rng_state=True)
        else:
            piece = _attention_block(q, key, value, valid, dropout)
        pieces.append(piece)
    return torch.cat(pieces, 2)


class MassFourier(nn.Module):
    def __init__(self, dimensions, frequencies):
        super().__init__()
        self.output_dim = dimensions * (1 + 2 * frequencies)
        self.register_buffer('frequencies', 2. ** torch.arange(frequencies, dtype=torch.float32))

    def forward(self, mass):
        with torch.autocast(device_type=mass.device.type, enabled=False):
            angle = mass.float()[..., None] * self.frequencies * math.pi
            return torch.cat((mass.float(), angle.sin().flatten(-2), angle.cos().flatten(-2)), -1)


class MassAttention(nn.Module):
    def __init__(self, cfg, rope=True):
        super().__init__()
        self.heads, self.width = cfg.heads, cfg.hidden // cfg.heads
        self.chunk, self.dropout, self.recompute = cfg.attention_chunk, cfg.dropout, cfg.checkpoint_attention
        self.rope = rope and cfg.use_rope
        if self.width % 2:
            raise ValueError('Attention head width must be even for mass RoPE')
        self.register_buffer('frequencies', torch.logspace(0., math.log10(cfg.rope_max_frequency), self.width//2) * math.pi)
        self.query = nn.Linear(cfg.hidden, cfg.hidden)
        self.key = nn.Linear(cfg.hidden, cfg.hidden)
        self.value = nn.Linear(cfg.hidden, cfg.hidden)
        self.output = nn.Linear(cfg.hidden, cfg.hidden)

    def rotate(self, value, positions):
        with torch.autocast(device_type=value.device.type, enabled=False):
            angle = positions.float()[:, None, :, None] * self.frequencies
            even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
            return torch.stack((even*angle.cos()-odd*angle.sin(), even*angle.sin()+odd*angle.cos()), -1).flatten(-2)

    def forward(self, query, tokens, valid, query_mass=None, token_mass=None):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(tokens)), split(self.value(tokens))
        if self.rope:
            if query_mass is None or token_mass is None:
                raise ValueError('Mass positions required for RoPE')
            q, k = self.rotate(q, query_mass), self.rotate(k, token_mass)
        result = stable_attention(q, k, v, valid, self.chunk,
                                  self.dropout if self.training else 0., self.recompute)
        return self.output(result.transpose(1, 2).reshape_as(query))


class SpectrumBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.hidden), nn.LayerNorm(cfg.hidden)
        self.attention = MassAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden*4), nn.GELU(), nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.hidden*4, cfg.hidden))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token, mass, valid):
        normalized = self.norm1(token)
        token = token + self.dropout(self.attention(normalized, normalized, valid, mass, mass))
        token = token + self.dropout(self.ff(self.norm2(token)))
        return token.masked_fill(~valid[..., None], 0.)


def region_bank(token, mass, ratio, intensity, visible, bins):
    """All visible peaks contribute; grid membership uses visible m/z only."""
    batch, peaks, hidden = token.shape
    safe_ratio = torch.where(visible, ratio, torch.zeros_like(ratio))
    index = (safe_ratio * bins).floor().long().clamp(0, bins-1)
    with torch.autocast(device_type=token.device.type, enabled=False):
        weight = torch.where(visible, intensity, torch.zeros_like(intensity)).float()
        total = weight.new_zeros(batch, bins).scatter_add(1, index, weight)
        values = token.float().masked_fill(~visible[..., None], 0.) * weight[..., None]
        pooled = token.new_zeros(batch, bins, hidden, dtype=torch.float32).scatter_add(1, index[..., None].expand(-1, -1, hidden), values)
        center = weight.new_zeros(batch, bins).scatter_add(1, index, torch.where(visible, mass, torch.zeros_like(mass)).float()*weight)
        return (pooled/total[..., None].clamp_min(1e-12)).to(token.dtype), center/total.clamp_min(1e-12), total > 0


class RoleCrossBlock(nn.Module):
    """Separate peak, mass-region and whole-spectrum memories, then learned mixing."""
    def __init__(self, cfg):
        super().__init__()
        self.local, self.region, self.global_attention = MassAttention(cfg), MassAttention(cfg), MassAttention(cfg, rope=False)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.hidden), nn.LayerNorm(cfg.hidden)
        self.gate = nn.Linear(cfg.hidden, 3)
        self.ff = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden*4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden*4, cfg.hidden))
        self.dropout = nn.Dropout(cfg.dropout)
        self.multiscale = cfg.use_multiscale

    def forward(self, node, mass, banks):
        raw, regions, global_token = banks
        query = self.norm1(node)
        values = [self.local(query, raw[0], raw[2], mass, raw[1]),
                  self.region(query, regions[0], regions[2], mass, regions[1]),
                  self.global_attention(query, global_token[0], global_token[1])]
        weights = self.gate(query).float().softmax(-1)
        enabled = weights.new_tensor([1., float(self.multiscale), 1.])
        weights = weights * enabled
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        update = sum(value*weights[..., i, None].to(value.dtype) for i, value in enumerate(values))
        node = node + self.dropout(update)
        return node + self.dropout(self.ff(self.norm2(node)))
