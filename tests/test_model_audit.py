"""Independent probability checks and regressions for mass-bin approximations."""
import math
import copy
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest('Install .[neural] for the model audit tests')
from casmi_neural.data import PrepareConfig,entry_record,collate
from casmi_neural.flow import action_probabilities,propagate,decompose,losses,reconstruction
from casmi_neural.mass_check import node_mass_bounds,check_peak
from test_neural import toy_entry


def flow_batch():
    # Root->A/B, both A/B->C, A->F and C->F. Paths merge and have unequal lengths.
    source=torch.tensor([0,0,1,2,1,3]);child=torch.tensor([1,2,3,3,4,4])
    return dict(edge_index=torch.stack((source,child)),roots=torch.tensor([0]),others=torch.tensor([4]),
                node_bin=torch.tensor([0,1,2,2,3]),node_batch=torch.zeros(5,dtype=torch.long),
                bin_batch=torch.zeros(5,dtype=torch.long),target=torch.tensor([0.,.1,.1,.7,.1]),num_graphs=1,
                edge_layers=[torch.tensor([0,1]),torch.tensor([2,3,4]),torch.tensor([5])])


def enumerate_paths(batch,stop,transition,other):
    edges=batch['edge_index'].T.tolist();result={('OTHER',):float(other)}
    def walk(node,probability,path,visited):
        result[path]=probability*float(stop[node])
        for e,(source,child) in enumerate(edges):
            if source==node:walk(child,probability*float(transition[e]),path+(child,),visited+(e,))
    walk(0,1-float(other),(0,),())
    return result


class ProbabilityAuditTests(unittest.TestCase):
    def test_path_kl_equals_independent_complete_path_enumeration(self):
        b=flow_batch();torch.manual_seed(18);flows=[];paths=[]
        for other in [.13,.27]:
            stop,edge=action_probabilities(torch.randn(5),torch.randn(6),b['edge_index'][0])
            flows.append(propagate(b,stop,edge,torch.tensor([other])))
            paths.append(enumerate_paths(b,stop,edge,other))
        q,p=paths
        exact=sum(value*math.log(value/p[key]) for key,value in q.items() if value>0)
        actual=float(losses(b,dict(posterior=flows[0],prior=flows[1]))['posterior_kl'])
        self.assertAlmostEqual(sum(q.values()),1.,places=6)
        self.assertAlmostEqual(actual,exact,places=6)

    def test_peak_edge_decomposition_equals_enumerated_path_contributions(self):
        b=flow_batch();torch.manual_seed(11)
        stop,edge=action_probabilities(torch.randn(5),torch.randn(6),b['edge_index'][0])
        flow=propagate(b,stop,edge,torch.tensor([.15]));parts=decompose(b,flow)
        expected=torch.zeros_like(parts['edge_by_bin']);edges=b['edge_index'].T.tolist()
        paths=enumerate_paths(b,stop,edge,.15)
        for path,probability in paths.items():
            if path==('OTHER',):continue
            column=int(b['node_bin'][path[-1]])
            for pair in zip(path[:-1],path[1:]):expected[edges.index(list(pair)),column]+=probability
        torch.testing.assert_close(parts['edge_by_bin'],expected)

    def test_reconstruction_cannot_identify_two_isobaric_formation_paths(self):
        # Two different intermediates lead to different fragment hypotheses
        # with the same terminal mass. Both paths have length two.
        b=dict(edge_index=torch.tensor([[0,0,1,2],[1,2,3,4]]),roots=torch.tensor([0]),others=torch.tensor([4]),
               node_bin=torch.tensor([0,1,2,3,3]),node_batch=torch.zeros(5,dtype=torch.long),
               bin_batch=torch.zeros(5,dtype=torch.long),target=torch.tensor([0.,0.,0.,1.,0.]),num_graphs=1,
               edge_layers=[torch.tensor([0,1]),torch.tensor([2,3])])
        outputs=[]
        for fraction in [.2,.8]:
            a=torch.tensor(fraction,requires_grad=True)
            flow=propagate(b,torch.tensor([0.,0.,0.,1.,1.]),torch.stack((a,1-a,a*0+1,a*0+1)),torch.tensor([0.]))
            value=reconstruction(b['target'],flow['spectrum'],b['bin_batch'],1).sum()
            self.assertAlmostEqual(float(value.detach()),0.,places=6)
            self.assertAlmostEqual(float(torch.autograd.grad(value,a)[0]),0.,places=6)
            outputs.append(flow)
        torch.testing.assert_close(outputs[0]['spectrum'],outputs[1]['spectrum'])
        self.assertGreater(float((outputs[0]['edge_flow']-outputs[1]['edge_flow']).abs().max().detach()),.5)


class MassAuditTests(unittest.TestCase):
    def record(self,mz):
        e=toy_entry();e['dag_nodes']=[e['dag_nodes'][1],e['dag_nodes'][0],e['dag_nodes'][3]]
        root,a,b=e['dag_nodes'];root.update(out_nids=[a['nid'],b['nid']]);a.update(mass=50.,in_nids=[root['nid']]);b.update(mass=50.006,in_nids=[root['nid']])
        e['predicted_ms2']=[[mz,1.]]
        return entry_record(e,'predicted',PrepareConfig())

    def test_bin_membership_does_not_imply_mass_compatibility(self):
        r=self.record(49.995);mass,bounds,source=node_mass_bounds(r)
        check=check_peak(mass,bounds,49.995,PrepareConfig())
        self.assertEqual(source,'exported_node_masses')
        self.assertEqual(int(r['node_bin'][1]),int(r['node_bin'][2]))
        self.assertEqual(check[1]['mass_compatibility'],'compatible')
        self.assertEqual(check[2]['mass_compatibility'],'incompatible')

    def test_anchor_can_miss_a_compatible_fragment(self):
        r=self.record(50.012);mass,bounds,_=node_mass_bounds(r)
        self.assertEqual(int(r['peak_bin'][0]),len(r['bin_masses']))
        self.assertEqual(check_peak(mass,bounds,50.012,PrepareConfig())[2]['mass_compatibility'],'compatible')

    def test_legacy_ratio_rounding_bounds_cover_exported_masses(self):
        r=self.record(49.995);actual=r.pop('node_masses');mass,bounds,source=node_mass_bounds(r)
        self.assertEqual(source,'cached_ratio_with_rounding_bound')
        self.assertTrue(((mass-actual).abs()<=bounds).all())
        check=check_peak(mass,bounds,49.995,PrepareConfig())
        self.assertEqual(check[2]['mass_compatibility'],'incompatible')

    def test_old_cache_boundary_is_uncertain_instead_of_falsely_exact(self):
        r=self.record(49.995);r.pop('node_masses');mass,bounds,_=node_mass_bounds(r)
        check=check_peak(mass,bounds,float(mass[0])+.007,PrepareConfig())
        self.assertEqual(check[0]['mass_compatibility'],'boundary_uncertain')

    def test_optional_exact_masses_are_diagnostics_not_new_model_inputs(self):
        from casmi_neural.model import ModelConfig,SpectrumDAGModel
        from casmi_neural.data import visibility
        r=self.record(49.995);changed=copy.deepcopy(r);changed['node_masses'].fill_(float('nan'))
        m=SpectrumDAGModel(ModelConfig(architecture='multiscale_v2',hidden=32,heads=4,dag_layers=1,
            spectrum_layers=1,cross_layers=1,dropout=0.,fourier_frequencies=4,region_bins=(2,4))).eval()
        with torch.no_grad():
            a=m(visibility(collate([r]),0.,0))['posterior']['spectrum']
            b=m(visibility(collate([changed]),0.,0))['posterior']['spectrum']
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
