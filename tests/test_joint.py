"""Scientific contracts and numerical checks for typed DAG joint retrieval."""
from dataclasses import asdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib
import torch
from test_peaks import fixture
from casmi_peaks.data import Catalog,PeakDataset,spectrum_batch,digest,signature
from casmi_peaks.infer import read_spectrum
from casmi_joint.model import JointConfig,JointAnnotator,RelationNetwork,joint_update
from casmi_joint.relations import GraphView,graph_record,make_training,RelationStore,TYPES
from casmi_joint.predict import rank_record
from casmi_joint.engine import train,read_config,checkpoint_guard
from casmi_joint.evaluate import evaluate_model,load_calibration


def relation_fixture(directory,data):
    directory=Path(directory); directory.mkdir()
    cat=Catalog(data); n=len(cat.labels)
    by_smiles={v['smiles']:v['id'] for v in cat.labels}
    graph=dict(nids=[100,50,900,3],labels=[by_smiles['CCO'],by_smiles['CC'],by_smiles['CO'],-1],
               children=[[1,2],[3],[],[]],probabilities=[.1]*n)
    with sqlite3.connect(directory/'relations.sqlite') as db:
        db.execute('CREATE TABLE graphs (key TEXT PRIMARY KEY,payload BLOB)')
        db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY,split TEXT,graph_key TEXT)')
        for split in ('train','val'):
            db.execute('INSERT INTO graphs VALUES (?,?)',(split,zlib.compress(json.dumps(graph).encode())))
        for row in map(json.loads,(Path(data)/'index.jsonl').read_text().splitlines()):
            if row['split']!='test': db.execute('INSERT INTO records VALUES (?,?,?)',(row['id'],row['split'],row['split']))
    meta=json.loads((Path(data)/'meta.json').read_text())
    rmeta=dict(schema_version=1,types=list(TYPES),peak_signature=signature(meta),partial=True,test_graphs_exported=0,
               hashes={'relations.sqlite':digest(directory/'relations.sqlite')})
    (directory/'meta.json').write_text(json.dumps(rmeta)); return rmeta


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.data=self.root/'data'; self.records=fixture(self.data); self.catalog=Catalog(self.data)
        self.relations=self.root/'relations'; self.rmeta=relation_fixture(self.relations,self.data)
        self.cfg=JointConfig(hidden=32,heads=4,spectrum_layers=2,graph_layers=2,mass_frequencies=8,relation_rank=8,dropout=0.)
        self.settings=dict(candidate_k=16,strength=2.,iterations=2,neighbors=3,support_k=4,damping=.5)

    def tearDown(self): self.temp.cleanup()

    def test_state_topology_direction_transitivity_and_duplicate_label_aggregation(self):
        view=GraphView(dict(labels=[0,1,2,1],children=[[1,2],[],[3],[]]))
        self.assertEqual(view.flags(0,1),[1.,1.,0.,1.])
        self.assertEqual(view.flags(1,0),[0.,0.,0.,1.])
        self.assertEqual(view.flags(1,2),[0.,0.,1.,1.])
        with self.assertRaisesRegex(ValueError,'Cycle'): GraphView(dict(labels=[0,1],children=[[1],[0]]))

    def test_sparse_nid_export_and_no_root_labels(self):
        from casmi_peaks.chem import structure
        key=structure('CCO','C2H5O','[M+H]+')['key']
        entry=dict(experiment_metadata={'precursor_type':'[M+H]+'},dag_nodes=[
            dict(nid=99,in_nids=[],out_nids=[7],SMILES='CCO',formula='C2H5O',final_prob=.1),
            dict(nid=7,in_nids=[99],out_nids=[],SMILES='CCO',formula='C2H5O',final_prob=.9)])
        graph=graph_record(entry,{key:3})
        self.assertEqual(graph['labels'],[-1,3]); self.assertEqual(graph['children'],[[1],[]])
        self.assertEqual(graph_record(entry,{})['labels'],[-1,-1])

    def test_no_dag_or_target_input_and_mass_pair_permutation_equivariance(self):
        torch.manual_seed(9); model=JointAnnotator(self.cfg).eval(); r=self.records[0]
        poison={**r,'dag_nodes':[{'nid':999}],'targets':[[['secret',1.]]]*3,'true_mol_metadata':{'SMILES':'CCC'}}
        self.assertEqual(read_spectrum(r),read_spectrum(poison))
        perm=[2,0,1]; shuffled={**r,'peaks':[r['peaks'][i] for i in perm]}
        with torch.no_grad():
            a=model.encode_spectrum(spectrum_batch([r])); b=model.encode_spectrum(spectrum_batch([poison]))
            c=model.encode_spectrum(spectrum_batch([shuffled]))
        self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.allclose(a[:,perm],c,atol=1e-5))
        x=rank_record(model,r,self.catalog,torch.device('cpu'),self.settings)
        y=rank_record(model,poison,self.catalog,torch.device('cpu'),self.settings)
        self.assertEqual(x['joint_rank'],y['joint_rank'])

    def test_joint_training_gradients_padding_and_no_support_are_finite(self):
        model=JointAnnotator(self.cfg); store=RelationStore(self.relations,json.loads((self.data/'meta.json').read_text()))
        t,r=make_training(self.records[:2],self.catalog,store,random.Random(0),8,4,4,'set')
        ids=t.pop('ids'); out=model(spectrum_batch(self.records[:2]),self.catalog.graphs(ids),t,r,
                                  {k:v for k,v in self.settings.items() if k!='candidate_k'})
        self.assertGreater(int((~t['supervised']).sum()),0)
        self.assertFalse(bool(t['positive'][~t['supervised']].any()))
        sum(out.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        store.close()
        unary=torch.tensor([[.1,0.],[0.,0.]],requires_grad=True); valid=torch.tensor([[True,True],[False,False]])
        emb=torch.randn(2,2,32,requires_grad=True)
        result=joint_update(unary,emb,torch.ones(2,2),valid,torch.tensor([10.,20.]),torch.tensor([[.3,1.,1.]]*2),model.relation)
        self.assertTrue(torch.isfinite(result[valid]).all())
        result[valid].sum().backward(); self.assertTrue(torch.isfinite(unary.grad).all())
        self.assertTrue(torch.equal(joint_update(unary,emb,torch.ones(2,2),valid,torch.ones(2),torch.ones(2,3),model.relation,strength=0),unary))
        ragged=torch.tensor([[True,True],[True,False]])
        result=joint_update(unary,emb,torch.ones(2,2),ragged,torch.tensor([10.,20.]),torch.tensor([[.3,1.,1.]]*2),model.relation,damping=1.)
        self.assertTrue(torch.isfinite(result[ragged]).all())

    def test_relation_evidence_can_change_a_peak_without_teacher_injection(self):
        class KnownRelation:
            rank=1
            def project(self,e):
                x=e[...,:1].unsqueeze(-2).expand(*e.shape[:-1],4,1); return x,x
            def logits(self,left,right,ma,mb,c): return (left*right).sum(-1)*5
            compatibility=staticmethod(RelationNetwork.compatibility)
        unary=torch.tensor([[.1,0.],[2.,-10.]])
        e=torch.tensor([[[-1.],[1.]],[[1.],[0.]]]); valid=torch.tensor([[True,True],[True,False]])
        score=joint_update(unary,e,torch.ones(2,2),valid,torch.tensor([10.,20.]),torch.tensor([[.3,1.,1.]]*2),
                           KnownRelation(),strength=2,iterations=2,neighbors=1,support_k=2)
        self.assertEqual(int(unary[0].argmax()),0); self.assertEqual(int(score[0].argmax()),1)

    def test_store_rejects_test_and_wrong_catalog(self):
        meta=json.loads((self.data/'meta.json').read_text()); store=RelationStore(self.relations,meta)
        with self.assertRaisesRegex(ValueError,'test'): store.get(self.records[-1])
        store.close()
        with self.assertRaisesRegex(ValueError,'signature'): RelationStore(self.relations,{**meta,'label_semantics':'different'})

    def test_store_rejects_exported_test_dags_even_with_updated_hash(self):
        with sqlite3.connect(self.relations/'relations.sqlite') as db:
            db.execute("INSERT INTO records VALUES (999,'test','val')")
        meta=json.loads((self.relations/'meta.json').read_text())
        meta['hashes']['relations.sqlite']=digest(self.relations/'relations.sqlite')
        (self.relations/'meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'test'): RelationStore(self.relations,json.loads((self.data/'meta.json').read_text()))

    def test_evaluation_oov_denominators_shortlist_bound_and_zero_strength(self):
        model=JointAnnotator(self.cfg).eval(); dataset=PeakDataset(self.data,['experimental'],'val')
        result=evaluate_model(model,dataset,self.catalog,torch.device('cpu'),self.settings,strengths=[0.,2.],catalog_batch=4)
        self.assertEqual(result['coverage']['teacher_annotated_peaks'],3)
        self.assertEqual(result['coverage']['vocab_covered_peaks'],2)
        self.assertEqual(result['joint_by_strength']['2']['teacher_top3_all_annotated'],2/3)
        self.assertEqual(result['joint_by_strength']['0']['teacher_top1_all_annotated'],result['unary']['teacher_top1_all_annotated'])
        self.assertLessEqual(result['joint_by_strength']['2']['teacher_top1_all_annotated'],result['coverage']['joint_shortlist_oracle_top1_all_annotated'])

    def test_checkpoint_and_calibration_identity_guards(self):
        meta=json.loads((self.data/'meta.json').read_text()); cfg={'model':asdict(self.cfg)}
        ckpt=dict(task='peak_fragment_joint_v2',dataset_signature=signature(meta),model_config=cfg['model'],relation_signature=self.rmeta)
        checkpoint_guard(ckpt,cfg,meta,self.rmeta)
        with self.assertRaisesRegex(ValueError,'Relation'): checkpoint_guard(ckpt,cfg,meta,{})
        path=self.root/'calibration.json'; path.write_text(json.dumps(dict(split='test',checkpoint_sha256='',base_settings=self.settings)))
        with self.assertRaisesRegex(ValueError,'split'): load_calibration(path,self.data/'meta.json',self.settings)

    def test_relation_audit_exercises_all_typed_heads_on_heldout_records(self):
        from casmi_joint.audit import audit_relations
        model=JointAnnotator(self.cfg); meta=json.loads((self.data/'meta.json').read_text())
        checkpoint=self.root/'audit.pt'
        torch.save(dict(task='peak_fragment_joint_v2',dataset_signature=signature(meta),model_config=asdict(self.cfg),
            relation_signature=self.rmeta,model=model.state_dict(),epoch=0),checkpoint)
        with redirect_stdout(io.StringIO()):
            result=audit_relations(checkpoint,self.data,self.relations,self.root/'audit.json','cpu')
        self.assertEqual(result['split'],'val'); self.assertGreater(result['sampled_pairs'],0)
        self.assertEqual(set(result['metrics']),set(TYPES))
        self.assertGreater(result['metrics']['common_precursor']['weak_negative_pairs'],0)
        with self.assertRaisesRegex(ValueError,'Test'): audit_relations(checkpoint,self.data,self.relations,self.root/'bad.json','cpu','test')

    def test_joint_epoch_resume_restores_rng_optimizer_and_scheduler(self):
        cfg=dict(seed=7,sources=['experimental'],loss_mode='set',model=asdict(self.cfg)|{'dropout':.1},joint=self.settings,
            training=dict(epochs=3,batch_size=2,eval_batch_size=3,learning_rate=.001,max_target_peaks=8,negatives=4,
                relation_pairs=4,relation_weight=.1,unary_weight=.2,cpu_threads=2,catalog_batch=4,workers=0))
        path=self.root/'config.json'; path.write_text(json.dumps(cfg))
        from casmi_joint import engine
        original=engine.evaluate_model; calls=[]
        def interrupt(*args,**kwargs):
            calls.append(1)
            if len(calls)==2: raise RuntimeError('simulated interruption')
            return original(*args,**kwargs)
        with redirect_stdout(io.StringIO()):
            train(path,self.data,self.relations,self.root/'full','cpu')
            with patch.object(engine,'evaluate_model',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated'): train(path,self.data,self.relations,self.root/'resumed','cpu')
            train(path,self.data,self.relations,self.root/'resumed','cpu',resume=self.root/'resumed/last.pt')
        a=torch.load(self.root/'full/last.pt',weights_only=True); b=torch.load(self.root/'resumed/last.pt',weights_only=True)
        self.assertEqual(a['scheduler'],b['scheduler']); self.assertEqual(a['global_step'],b['global_step'])
        for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)


if __name__=='__main__': unittest.main()
