import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest('Install .[neural] for v2 tests')
from torch import nn
from casmi_neural.attention import stable_attention, MassAttention, region_bank
from casmi_neural.model import ModelConfig, SpectrumDAGModel, normalized_model_config
from casmi_neural.data import PrepareConfig, collate, visibility, entry_record
from casmi_neural.flow import losses, decompose
from test_neural import toy_entry, batch

torch.set_num_threads(1)


def config(**overrides):
    values=dict(architecture='multiscale_v2', hidden=32, heads=4, dag_layers=1,
                spectrum_layers=1, cross_layers=1, dropout=0., fourier_frequencies=4,
                region_bins=(2,4), attention_chunk=2)
    values.update(overrides)
    return ModelConfig(**values)


def model(**overrides):
    result=SpectrumDAGModel(config(**overrides))
    # Activate spectrum dependence so leakage/isolation checks cannot pass just
    # because the production residual heads deliberately start at zero.
    for module in [result.v2.posterior_head.stop, result.v2.posterior_head.edge[-1], result.v2.posterior_head.other]:
        nn.init.normal_(module.weight, std=.08)
    return result


class StableAttentionTests(unittest.TestCase):
    def test_chunked_and_recomputed_attention_match_dense_math_and_gradients(self):
        torch.manual_seed(9)
        inputs=[torch.randn(2,2,n,4,requires_grad=True) for n in [5,7,7]]
        valid=torch.tensor([[1,1,0,1,0,1,1],[0,1,1,1,1,0,0]],dtype=torch.bool)
        ref=(inputs[0]@inputs[1].transpose(-2,-1)/2).masked_fill(~valid[:,None,None],-torch.inf).softmax(-1)@inputs[2]
        expected=torch.autograd.grad(ref.square().sum(), inputs)
        for recompute in [False,True]:
            values=[x.detach().clone().requires_grad_() for x in inputs]
            result=stable_attention(*values,valid,chunk_size=2,recompute=recompute)
            torch.testing.assert_close(result,ref)
            grads=torch.autograd.grad(result.square().sum(), values)
            for a,b in zip(grads,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_entirely_hidden_memory_has_zero_finite_output_and_gradients(self):
        q=torch.randn(1,2,3,4,requires_grad=True); k=torch.randn(1,2,2,4,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
        out=stable_attention(q,k,v,torch.zeros(1,2,dtype=torch.bool),1,recompute=True)
        self.assertEqual(float(out.detach().abs().sum()),0.)
        out.sum().backward()
        for value in [q,k,v]:self.assertTrue(torch.isfinite(value.grad).all())

    def test_mass_rotations_preserve_relative_displacement(self):
        attention=MassAttention(config())
        q=torch.randn(1,4,3,8); k=torch.randn(1,4,4,8)
        qp=torch.tensor([[.1,.2,.4]]);kp=torch.tensor([[.15,.3,.5,.7]])
        first=attention.rotate(q,qp)@attention.rotate(k,kp).transpose(-2,-1)
        shifted=attention.rotate(q,qp+.25)@attention.rotate(k,kp+.25).transpose(-2,-1)
        torch.testing.assert_close(first,shifted,atol=.03,rtol=.003)

    def test_region_pooling_ignores_hidden_values_and_is_peak_order_invariant(self):
        token=torch.randn(1,4,8);mass=torch.tensor([[.1,.2,.3,.4]]);ratio=mass.clone()
        intensity=torch.tensor([[.1,.2,.3,.4]]);visible=torch.tensor([[1,0,1,0]],dtype=torch.bool)
        expected=region_bank(token,mass,ratio,intensity,visible,4)
        token[:,~visible[0]]=float('nan');mass[~visible]=float('nan');ratio[~visible]=float('nan');intensity[~visible]=float('nan')
        actual=region_bank(token,mass,ratio,intensity,visible,4)
        order=torch.tensor([3,1,0,2])
        permuted=region_bank(token[:,order],mass[:,order],ratio[:,order],intensity[:,order],visible[:,order],4)
        for a,b,c in zip(expected,actual,permuted):
            torch.testing.assert_close(a,b);torch.testing.assert_close(a,c)


class V2ModelTests(unittest.TestCase):
    def test_forward_backward_avoids_fused_sdpa_and_has_all_parameter_gradients(self):
        m=model();b=visibility(batch(),.5,1)
        with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=AssertionError('Fused SDPA forbidden')):
            out=m(b);losses(b,out)['loss'].mean().backward()
        for name,value in m.named_parameters():
            self.assertIsNotNone(value.grad,name)
            self.assertTrue(torch.isfinite(value.grad).all(),name)
        self.assertAlmostEqual(float(out['posterior']['spectrum'].detach().sum()),1.,places=6)
        parts=decompose(b,out['posterior'])
        torch.testing.assert_close(parts['edge_by_bin'].sum(1),out['posterior']['edge_flow'])

    def test_masked_mass_intensity_and_target_do_not_leak_with_active_residuals(self):
        m=model().eval();a=visibility(batch(),.5,42);b=copy.deepcopy(a)
        b['tokens'][~b['visible']]=float('nan');b['target']=torch.rand_like(b['target'])
        with torch.no_grad():
            x=m(a)['posterior']['spectrum'];y=m(b)['posterior']['spectrum']
        torch.testing.assert_close(x,y)
        c=copy.deepcopy(a);c['tokens'][c['visible']]=torch.tensor([.9,.5])
        with torch.no_grad():z=m(c)['posterior']['spectrum']
        self.assertGreater(float((x-z).abs().max()),1e-6)

    def test_entire_spectrum_hidden_is_finite(self):
        b=batch();b['visible']=torch.zeros_like(b['token_valid']);b['tokens'].fill_(float('nan'))
        out=model()(b)
        self.assertTrue(torch.isfinite(out['posterior']['spectrum']).all())
        losses(b,out)['loss'].mean().backward()

    def test_bfloat16_encoder_keeps_flow_and_losses_in_float32(self):
        m=model();b=visibility(batch(),.5,1)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=m(b);value=losses(b,out)['loss'].mean()
        self.assertEqual(out['posterior']['spectrum'].dtype,torch.float32)
        self.assertEqual(value.dtype,torch.float32)
        value.backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_batched_graphs_with_different_peak_counts_are_isolated(self):
        m=model().eval();a=entry_record(toy_entry(),'predicted',PrepareConfig())
        e=toy_entry();e['predicted_ms2']=[[30.,1.],[45.,2.],[70.,3.],[89.,2.]]
        c=entry_record(e,'predicted',PrepareConfig())
        with torch.no_grad():
            joined=m(visibility(collate([a,c]),0.,0))['posterior']['spectrum']
            separate=torch.cat([m(visibility(collate([r]),0.,0))['posterior']['spectrum'] for r in [a,c]])
        torch.testing.assert_close(joined,separate,atol=2e-6,rtol=2e-5)

    def test_graph_and_peak_permutations_preserve_outputs(self):
        m=model().eval();record=entry_record(toy_entry(),'predicted',PrepareConfig())
        e=toy_entry();e['dag_nodes']=e['dag_nodes'][::-1]
        permuted=entry_record(e,'predicted',PrepareConfig())
        permuted['peaks']=permuted['peaks'].flip(0);permuted['peak_bin']=permuted['peak_bin'].flip(0)
        with torch.no_grad():
            a=m(visibility(collate([record]),0.,0))['posterior']['spectrum']
            b=m(visibility(collate([permuted]),0.,0))['posterior']['spectrum']
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_frozen_graph_prior_remains_fixed_with_dropout(self):
        m=model(dropout=.4);m.freeze_graph_prior();m.train();b=visibility(batch(),0.,0)
        a=m(b)['prior']['spectrum'];c=m(b)['prior']['spectrum']
        torch.testing.assert_close(a,c);self.assertFalse(a.requires_grad)
        losses(b,m(b))['loss'].mean().backward()
        self.assertTrue(any(p.grad is not None for p in m.v2.cross.parameters()))

    def test_single_node_graph_and_ablations_keep_ddp_parameters_connected(self):
        e=toy_entry();e['dag_nodes']=[e['dag_nodes'][1]];e['dag_nodes'][0].update(in_nids=[],out_nids=[])
        e['predicted_ms2']=[[100.,1.],[89.,.1]]
        b=visibility(collate([entry_record(e,'predicted',PrepareConfig())]),0.,0)
        for options in [{},{'use_multiscale':False,'use_rope':False},{'dag_attention':False},{'use_spectrum':False}]:
            m=model(**options);losses(b,m(b))['loss'].mean().backward()
            for name,value in m.named_parameters():
                self.assertIsNotNone(value.grad,name)
                self.assertTrue(torch.isfinite(value.grad).all(),name)

    def test_legacy_checkpoint_keys_and_config_are_backward_compatible(self):
        old=dict(hidden=24,heads=4,dag_layers=1,spectrum_layers=1,cross_layers=1,dropout=0.,use_fingerprints=True,use_spectrum=True)
        self.assertEqual(normalized_model_config(old),asdict(ModelConfig(**old)))
        m=SpectrumDAGModel(ModelConfig(**old));self.assertFalse(any(k.startswith('v2.') for k in m.state_dict()))
        loaded=SpectrumDAGModel(ModelConfig(**old));loaded.load_state_dict(m.state_dict())


if __name__=='__main__':unittest.main()
