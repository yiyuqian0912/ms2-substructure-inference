"""Differentiable DAG probability flow and exact peak-conditioned decomposition.

A state either halts or transitions to one child. Upstream edges can support
multiple peaks. This is signal attribution under a model, not proven chemistry.
"""
import torch


def segment_sum(x, index, size):
    return x.new_zeros((size,) + x.shape[1:]).index_add(0,index,x)


def action_probabilities(stop_logits, edge_logits, source):
    # Run in FP32 even with BF16 encoders. Every node has a stop action; a leaf
    # therefore halts with probability exactly 1, preserving total flow.
    stop_logits, edge_logits = stop_logits.float(), edge_logits.float()
    ids = torch.cat([torch.arange(len(stop_logits),device=source.device),source])
    values = torch.cat([stop_logits,edge_logits])
    maximum = torch.full_like(stop_logits,-torch.inf).scatter_reduce(0,ids,values,reduce='amax',include_self=True)
    exp = (values - maximum[ids].detach()).exp()
    denominator = segment_sum(exp,ids,len(stop_logits))
    probs = exp / denominator[ids]
    return probs[:len(stop_logits)], probs[len(stop_logits):]


def propagate(batch, stop, transition, other):
    source, child = batch['edge_index']
    reach = stop.new_zeros(len(stop)).index_add(0,batch['roots'],1-other)
    for edge_ids in batch['edge_layers']:
        reach = reach.index_add(0,child[edge_ids],reach[source[edge_ids]] * transition[edge_ids])
    emission = reach * stop
    spectrum = segment_sum(emission,batch['node_bin'],len(batch['target']))
    spectrum = spectrum.index_add(0,batch['others'],other)
    return {'spectrum':spectrum,'reach':reach,'emission':emission,
            'edge_flow':reach[source]*transition,'stop':stop,'transition':transition,'other':other}


def reconstruction(target,predicted,bin_batch,batch_size):
    p = predicted.float().clamp_min(1e-12)
    y = target.float()
    return segment_sum(y*(y.clamp_min(1e-12).log()-p.log()),bin_batch,batch_size)


def losses(batch,output,prior_weight=.3,kl_weight=.02,path_weight=.0001):
    q,p = output['posterior'],output['prior']; size=batch['num_graphs']
    rec = reconstruction(batch['target'],q['spectrum'],batch['bin_batch'],size)
    prior = reconstruction(batch['target'],p['spectrum'],batch['bin_batch'],size)
    source = batch['edge_index'][0]
    node_kl = q['stop'] * (q['stop'].clamp_min(1e-12).log()-p['stop'].clamp_min(1e-12).log())
    edge_kl = q['transition'] * (q['transition'].clamp_min(1e-12).log()-p['transition'].clamp_min(1e-12).log())
    node_kl = node_kl + segment_sum(edge_kl,source,len(node_kl))
    reach = q['reach']
    regularizer = segment_sum(reach*node_kl,batch['node_batch'],size)
    qo,po=q['other'].clamp(1e-7,1-1e-7),p['other'].clamp(1e-7,1-1e-7)
    regularizer = regularizer + qo*(qo.log()-po.log())+(1-qo)*((1-qo).log()-(1-po).log())
    path = segment_sum(q['edge_flow'],batch['node_batch'][source],size)
    total = rec+prior_weight*prior+kl_weight*regularizer+path_weight*path
    return {'loss':total,'reconstruction':rec,'prior_reconstruction':prior,
            'posterior_kl':regularizer,'expected_path_length':path}


def cosine_per_graph(target,prediction,groups,size):
    numerator=segment_sum(target*prediction,groups,size)
    denominator=(segment_sum(target.square(),groups,size)*segment_sum(prediction.square(),groups,size)).sqrt()
    return numerator/denominator.clamp_min(1e-12)


@torch.no_grad()
def decompose(batch,flow,bins=None):
    """Single-graph backward absorption probabilities for each requested mass bin.

    edge_by_bin[e,k] is UNNORMALIZED signal flowing through edge e and
    eventually stopping in bin k. Summing all bins recovers edge_flow[e].
    """
    if batch['num_graphs'] != 1:
        raise ValueError('Decompose one DAG at a time')
    if bins is None:
        bins=torch.arange(int(batch['others'][0]),device=batch['target'].device)
    source,child=batch['edge_index']; n=len(flow['stop'])
    h=(batch['node_bin'][:,None]==bins[None,:]).to(flow['stop'].dtype)*flow['stop'][:,None]
    for ids in reversed(batch['edge_layers']):
        h=h.index_add(0,source[ids],flow['transition'][ids,None]*h[child[ids]])
    edge_by_bin=flow['edge_flow'][:,None]*h[child]
    node_by_bin=flow['reach'][:,None]*h
    terminal_by_bin=flow['emission'][:,None]*(batch['node_bin'][:,None]==bins[None,:])
    return {'bins':bins,'edge_by_bin':edge_by_bin,'node_by_bin':node_by_bin,
            'terminal_by_bin':terminal_by_bin}
