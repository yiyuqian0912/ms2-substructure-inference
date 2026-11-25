"""Audit invariants: masks must not alter chemistry scores or their gradients."""
from dataclasses import replace
import math
import random
from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn.functional as F
from casmi_joint.model import JointConfig,JointAnnotator,RelationNetwork,joint_update,neighbor_indices


class KnownRelation:
    rank=1

    def project(self,embedding):
        x=embedding[...,:1].unsqueeze(-2).expand(*embedding.shape[:-1],4,1)
        return x,x

    def logits(self,left,right,ma,mb,c):
        return (left*right).sum(-1)*5

    compatibility=staticmethod(RelationNetwork.compatibility)


class JointModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_padding_does_not_change_valid_scores_or_gradients(self):
        def run(extra):
            unary=torch.tensor([[.02,0.],[0.,0.],[0.,0.]],requires_grad=True)
            embeddings=torch.tensor([[[-1.],[1.]],[[1.],[.8]],[[.6],[1.]]],requires_grad=True)
            u=F.pad(unary,(0,extra)); e=F.pad(embeddings,(0,0,0,extra))
            valid=F.pad(torch.ones(3,2,dtype=torch.bool),(0,extra))
            mass=F.pad(torch.ones(3,2),(0,extra))
            scores=joint_update(u,e,mass,valid,torch.tensor([10.,20.,30.]),torch.tensor([[.3,1.,1.]]*3),
                KnownRelation(),strength=2,iterations=2,neighbors=2,support_k=16,damping=.5)
            loss=(scores[:,:2]*torch.tensor([[.1,.8],[.4,.3],[.6,.2]])).sum()
            grads=torch.autograd.grad(loss,(unary,embeddings))
            return scores[:,:2].detach(),grads
        a,ga=run(0); b,gb=run(14)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        for x,y in zip(ga,gb): self.assertTrue(torch.allclose(x,y,atol=2e-6))

    def test_factorized_pair_contraction_matches_broadcast_scores_and_gradients(self):
        torch.manual_seed(11)
        cfg=JointConfig(hidden=32,heads=4,relation_rank=8)
        relation=RelationNetwork(cfg)
        left=torch.randn(2,3,4,8,requires_grad=True); right=torch.randn(2,5,4,8,requires_grad=True)
        ma=torch.rand(2,3)*100; mb=torch.rand(2,5)*100; c=torch.tensor([[.3,.5,1.],[.6,.8,-1.]])
        new=relation.pair_logits(left,right,ma,mb,c)
        expected=(left[:,:,None]*right[:,None]).sum(-1)/math.sqrt(cfg.relation_rank)
        shape=(2,3,5)
        expected=expected+relation.context_features(ma[:,:,None].expand(shape),mb[:,None].expand(shape),c[:,None,None].expand(2,3,5,3))
        self.assertTrue(torch.allclose(new,expected,atol=1e-6))
        a=torch.autograd.grad(new.square().sum(),(left,right),retain_graph=True)
        b=torch.autograd.grad(expected.square().sum(),(left,right))
        for x,y in zip(a,b): self.assertTrue(torch.allclose(x,y,atol=2e-6))

    def test_global_support_anchors_span_mass_range(self):
        masses=torch.arange(100,dtype=torch.float32)
        neighbors=neighbor_indices(masses,torch.ones(100,dtype=torch.bool),6)
        self.assertIn(0,neighbors[50].tolist()); self.assertIn(99,neighbors[50].tolist())
        self.assertIn(99,neighbors[0].tolist()); self.assertIn(0,neighbors[99].tolist())
        self.assertTrue(all(len(set(row))==6 for row in neighbors.tolist()))
        for i,row in enumerate(neighbors.tolist()): self.assertNotIn(i,row)

    def test_sender_mask_excludes_forced_labels_and_allows_one_sender(self):
        unary=torch.tensor([[0.,0.],[100.,0.]],requires_grad=True)
        embeddings=torch.tensor([[[-1.],[1.]],[[1.],[-1.]]])
        valid=torch.ones(2,2,dtype=torch.bool)
        support=torch.tensor([[False,False],[False,True]])
        kwargs=dict(strength=2,iterations=2,neighbors=1,support_k=2,damping=1.)
        args=(embeddings,torch.ones(2,2),valid,torch.tensor([10.,20.]),torch.tensor([[.3,1.,1.]]*2),KnownRelation())
        score=joint_update(unary,*args,support_valid=support,**kwargs)
        unrestricted=joint_update(unary,*args,**kwargs)
        self.assertEqual(int(score[0].argmax()),0)
        self.assertEqual(int(unrestricted[0].argmax()),1)
        poisoned=unary.detach().clone(); poisoned[1,0]=10000.
        repeated=joint_update(poisoned,*args,support_valid=support,**kwargs)
        self.assertTrue(torch.allclose(score[0],repeated[0],atol=1e-6))
        grad=torch.autograd.grad(score[0,0],unary)[0]
        self.assertEqual(float(grad[1,0]),0.)

    def test_empty_spectrum_and_nonfinite_hyperparameters_are_rejected(self):
        cfg=JointConfig(hidden=32,heads=4,spectrum_layers=1,graph_layers=1,mass_frequencies=4,relation_rank=4)
        model=JointAnnotator(cfg)
        for n in (0,2):
            with self.assertRaisesRegex(ValueError,'Empty'):
                model.encode_spectrum(dict(peaks=torch.zeros(1,n,2),valid=torch.zeros(1,n,dtype=torch.bool),conditions=torch.tensor([[.3,.5,1.]])))
        for name,value in [('temperature',float('nan')),('mass_tolerance',float('inf')),('dropout',float('nan')),('heads',0)]:
            with self.assertRaises(ValueError): replace(cfg,**{name:value})
        for changes in ({'strength':float('nan')},{'damping':float('inf')},{'neighbors':0},{'support_k':0}):
            with self.assertRaises(ValueError):
                joint_update(torch.zeros(2,2),torch.zeros(2,2,32),torch.ones(2,2),torch.ones(2,2,dtype=torch.bool),
                             torch.ones(2),torch.ones(2,3),model.relation,**changes)

    def test_peak_only_ablation_cannot_read_other_peaks(self):
        torch.manual_seed(23)
        cfg=JointConfig(hidden=32,heads=4,spectrum_layers=2,graph_layers=1,mass_frequencies=4,relation_rank=4,dropout=0.,peak_only=True)
        model=JointAnnotator(cfg).eval()
        batch=dict(peaks=torch.tensor([[[10.,100.],[20.,25.],[30.,10.],[0.,0.]]]),
            valid=torch.tensor([[True,True,True,False]]),conditions=torch.tensor([[.3,.1,1.]]))
        changed={**batch,'peaks':batch['peaks'].clone()}
        changed['peaks'][0,2]=torch.tensor([77.,80.])  # Base peak stays at 100.
        a=model.encode_spectrum(batch); b=model.encode_spectrum(changed)
        self.assertTrue(torch.isfinite(a).all()); self.assertTrue(torch.isfinite(b).all())
        self.assertTrue(torch.equal(a[0,1],b[0,1]))
        a[0,1].sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pair.parameters()))

    def test_disabled_pair_bias_keeps_parameter_graph_but_zero_gradient(self):
        model=JointAnnotator(JointConfig(hidden=32,heads=4,spectrum_layers=1,graph_layers=1,mass_frequencies=4,relation_rank=4,
                                        dropout=0.,use_pair_bias=False))
        batch=dict(peaks=torch.tensor([[[10.,100.],[20.,25.]]]),valid=torch.ones(1,2,dtype=torch.bool),conditions=torch.tensor([[.3,.1,1.]]))
        model.encode_spectrum(batch)[0,0].sum().backward()
        for p in model.pair.parameters():
            self.assertIsNotNone(p.grad); self.assertTrue(torch.equal(p.grad,torch.zeros_like(p.grad)))

    def test_per_head_weak_negative_weights_match_effective_bce(self):
        from test_peaks import fixture
        from test_joint import relation_fixture
        from casmi_peaks.data import Catalog,spectrum_batch
        from casmi_joint.relations import RelationStore,make_training
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); records=fixture(root/'data')[:2]; catalog=Catalog(root/'data')
            relation_fixture(root/'relations',root/'data')
            store=RelationStore(root/'relations',json.loads((root/'data/meta.json').read_text()))
            targets,relations=make_training(records,catalog,store,random.Random(2),8,4,4,'set')
            ids=targets.pop('ids'); graphs=catalog.graphs(ids)
            model=JointAnnotator(JointConfig(hidden=32,heads=4,spectrum_layers=1,graph_layers=1,mass_frequencies=4,relation_rank=4,dropout=0.))
            relations['head_weights']=torch.where(relations['labels']>0,1.,.1)
            out=model(spectrum_batch(records),graphs,targets,relations,dict(strength=0))
            f=model.fragment(graphs); left,right=model.relation.project(f)
            r=relations
            logits=model.relation.logits(left[r['left']],right[r['right']],r['ma'],r['mb'],r['conditions'])
            effective=r['weights'][:,None]*r['head_weights']
            expected=(F.binary_cross_entropy_with_logits(logits,r['labels'],reduction='none')*effective).sum()/effective.sum()
            self.assertTrue(torch.allclose(out['relation_loss'],expected,atol=1e-7))
            store.close()


if __name__=='__main__': unittest.main()
