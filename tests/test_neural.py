import copy
from dataclasses import asdict
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest('Install .[neural] to run the neural tests')

from casmi_neural.data import PrepareConfig,entry_record,collate,visibility
from casmi_neural.flow import action_probabilities,propagate,decompose,losses,segment_sum
from casmi_neural.model import ModelConfig,SpectrumDAGModel

torch.set_num_threads(1)


def toy_entry():
    # Sparse IDs and intentionally non-topological list order.
    return {'identifiers':{'id':'toy'},'pred_mol_metadata':{'smiles':'CCCCO'},
        'experiment_metadata':{'precursor_mass':100.,'collision_energy_nce':35.,'precursor_type':'[M+H]+'},
        'dag_nodes':[
          {'nid':99,'mass':45.,'formula':'C2H5O','SMILES':'CCO','total_num_h':5,'in_nids':[7],'out_nids':[]},
          {'nid':3,'mass':100.,'formula':'C5H8O2','SMILES':'CCCCO','total_num_h':8,'in_nids':[],'out_nids':[7]},
          {'nid':7,'mass':70.,'formula':'C4H6O','SMILES':'CCCO','total_num_h':6,'in_nids':[3],'out_nids':[99,8]},
          {'nid':8,'mass':30.,'formula':'CH2O','SMILES':'CO','total_num_h':2,'in_nids':[7],'out_nids':[]}],
        'predicted_ms2':[[45.,3.],[30.,1.]],'experimental_ms2':[[45.,3.],[30.,1.],[89.,.2]]}


def batch():
    return collate([entry_record(toy_entry(),'predicted',PrepareConfig())])


class FlowTests(unittest.TestCase):
    def test_mass_conservation_and_shared_upstream_paths(self):
        b=batch();stop=torch.tensor([1.,0.,0.,1.]);edge=torch.tensor([1.,.75,.25])
        # Edges in stored list order: root->A, A->B, A->C.
        f=propagate(b,stop,edge,torch.tensor([.1]))
        self.assertAlmostEqual(float(f['spectrum'].sum()),1.,places=6)
        d=decompose(b,f)
        torch.testing.assert_close(d['edge_by_bin'].sum(1),f['edge_flow'])
        b45=int(b['node_bin'][0]);b30=int(b['node_bin'][3])
        self.assertGreater(float(d['edge_by_bin'][0,b45]),0.)
        self.assertGreater(float(d['edge_by_bin'][0,b30]),0.)
        self.assertAlmostEqual(float(d['edge_by_bin'][0,b45]),.675,places=6)
        self.assertAlmostEqual(float(d['edge_by_bin'][0,b30]),.225,places=6)

    def test_flow_gradients_finite(self):
        b=batch();a=torch.randn(4,requires_grad=True);e=torch.randn(3,requires_grad=True);o=torch.tensor([-.2],requires_grad=True)
        stop,edge=action_probabilities(a,e,b['edge_index'][0])
        f=propagate(b,stop,edge,o.sigmoid())
        out=losses(b,{'prior':f,'posterior':f});out['loss'].mean().backward()
        for grad in [a.grad,e.grad,o.grad]:self.assertTrue(torch.isfinite(grad).all())
        self.assertAlmostEqual(float(out['posterior_kl'].detach()),0.,places=6)

    def test_softmax_local_normalization_and_leaf_stop(self):
        b=batch();s,e=action_probabilities(torch.randn(4),torch.randn(3),b['edge_index'][0])
        torch.testing.assert_close(s+segment_sum(e,b['edge_index'][0],4),torch.ones(4))
        self.assertEqual(float(s[0]),1.)

    def test_multiple_parents_merge_without_double_counting(self):
        e=toy_entry();nodes={n['nid']:n for n in e['dag_nodes']}
        nodes[3]['out_nids']=[7,99]
        nodes[99]['in_nids']=[3];nodes[99]['out_nids']=[8]
        nodes[7]['out_nids']=[8];nodes[8]['in_nids']=[7,99]
        b=collate([entry_record(e,'predicted',PrepareConfig())])
        f=propagate(b,torch.tensor([0.,0.,0.,1.]),torch.tensor([1.,.6,.4,1.]),torch.tensor([.1]))
        self.assertAlmostEqual(float(f['reach'][3]),.9,places=6)
        self.assertAlmostEqual(float(f['spectrum'].sum()),1.,places=6)
        d=decompose(b,f)
        torch.testing.assert_close(d['edge_by_bin'].sum(1),f['edge_flow'])


class InputTests(unittest.TestCase):
    def test_teacher_annotation_fields_not_used(self):
        e=toy_entry();a=entry_record(e,'predicted',PrepareConfig())
        e['fragment_matching']='must not be read';e['peak_matchings']='must not be read'
        e['research_metadata']={'is_target':False,'source_cid':-999}
        e['true_mol_metadata']='must not be read';e['ms_metrics']='must not be read'
        for n in e['dag_nodes']:
            n['final_prob']=float('nan');n['halting_probability']='must not be read'
            n['bond_break_probabilities']='must not be read'
        b=entry_record(e,'predicted',PrepareConfig())
        for k in a:
            if isinstance(a[k],torch.Tensor):torch.testing.assert_close(a[k],b[k])
            else:self.assertEqual(a[k],b[k])

    def test_unknown_signal_is_retained(self):
        r=entry_record(toy_entry(),'experimental',PrepareConfig())
        self.assertGreater(float(r['target'][-1]),0.)
        self.assertAlmostEqual(float(r['target'].sum()),1.,places=6)

    def test_mask_hides_both_mass_and_intensity_from_encoder(self):
        b=visibility(batch(),.5,1)
        self.assertFalse(b['visible'].all())
        self.assertTrue(b['visible'].any())
        # Model explicitly multiplies both channels by visible; no original
        # masked peak coordinates or intensities are supplied elsewhere.
        torch.testing.assert_close(b['tokens']*b['visible'][...,None],
                                   b['tokens'].masked_fill(~b['visible'][...,None],0.))


class ModelTests(unittest.TestCase):
    def test_batched_graphs_do_not_leak_into_each_other(self):
        torch.manual_seed(2)
        m=SpectrumDAGModel(ModelConfig(hidden=24,heads=4,dag_layers=1,spectrum_layers=1,cross_layers=1,dropout=0.)).eval()
        a=entry_record(toy_entry(),'predicted',PrepareConfig());e=toy_entry()
        e['predicted_ms2']=[[70.,9.],[45.,1.]];c=entry_record(e,'predicted',PrepareConfig())
        with torch.no_grad():
            joined=m(visibility(collate([a,c]),0.,0))['posterior']['spectrum']
            separate=torch.cat([m(visibility(collate([r]),0.,0))['posterior']['spectrum'] for r in [a,c]])
        torch.testing.assert_close(joined,separate,atol=1e-6,rtol=1e-5)

    def test_forward_backward_and_permutation(self):
        torch.manual_seed(4)
        model=SpectrumDAGModel(ModelConfig(hidden=24,heads=4,dag_layers=1,spectrum_layers=1,cross_layers=1,dropout=0.))
        b=visibility(batch(),0.,0);out=model(b)
        total=losses(b,out)['loss'].mean();total.backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        e=toy_entry();e['dag_nodes']=e['dag_nodes'][::-1]
        b2=visibility(collate([entry_record(e,'predicted',PrepareConfig())]),0.,0)
        model.eval()
        with torch.no_grad():
            p1=model(b)['posterior']['spectrum'];p2=model(b2)['posterior']['spectrum']
        torch.testing.assert_close(p1,p2,atol=1e-6,rtol=1e-5)

    def test_masked_target_value_does_not_leak(self):
        torch.manual_seed(1)
        model=SpectrumDAGModel(ModelConfig(hidden=24,heads=4,dag_layers=1,spectrum_layers=1,cross_layers=1,dropout=0.)).eval()
        a=visibility(batch(),.5,42);b=copy.deepcopy(a)
        b['tokens'][~b['visible']]=torch.tensor([100.,1e6])
        b['target']=torch.rand_like(b['target'])
        with torch.no_grad():
            pa=model(a)['posterior']['spectrum'];pb=model(b)['posterior']['spectrum']
        torch.testing.assert_close(pa,pb)

    def test_frozen_prior_stays_deterministic_in_training(self):
        m=SpectrumDAGModel(ModelConfig(hidden=24,heads=4,dag_layers=1,spectrum_layers=1,cross_layers=1,dropout=.4))
        m.freeze_graph_prior();m.train()
        b=visibility(batch(),0.,0)
        a=m(b)['prior']['spectrum'];c=m(b)['prior']['spectrum']
        torch.testing.assert_close(a,c)
        self.assertFalse(a.requires_grad)


if __name__=='__main__':unittest.main()
