"""Scientific/data contracts for the DAG-free peak annotator."""
from dataclasses import asdict
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

import torch
from casmi_peaks.chem import structure,graph_features
from casmi_peaks.data import Catalog,PeakDataset,PrepareConfig,_query,digest,spectrum_batch,training_targets
from casmi_peaks.model import ModelConfig,PeakAnnotator,retrieval_loss
from casmi_peaks.engine import evaluate_model,checkpoint_guard
from casmi_peaks.infer import read_spectrum


def fixture(directory):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    labels=[]
    for smiles,formula,mz in [('CCO','C2H5O',45.03349),('COC','C2H5O',45.03349),('CC','C2H5',29.03858),('CO','CH3O',31.01784)]:
        labels.append({**structure(smiles,formula,'[M+H]+'),'mz':mz,'mz_min':mz,'mz_max':mz,'mass_count':1,'train_peak_occurrences':2})
    labels.sort(key=lambda x:x['key'])
    for i,label in enumerate(labels):label['id']=i
    (directory/'vocab.json').write_text(json.dumps(labels))
    atoms=[];edges=[];bonds=[];numeric=[];n=[0];e=[0]
    for label in labels:
        a,b,c,d=graph_features(label);atoms.append(a);edges.append(b);bonds.append(c);numeric.append(d)
        n.append(n[-1]+len(a));e.append(e[-1]+b.shape[1])
    torch.save(dict(atoms=torch.cat(atoms),edges=torch.cat(edges,1),bonds=torch.cat(bonds),numeric=torch.stack(numeric),
        node_ptr=torch.tensor(n),edge_ptr=torch.tensor(e)),directory/'features.pt')
    by_smiles={v['smiles']:v for v in labels}
    unknown=structure('C1CO1','C2H5O','[M+H]+')['key']
    def record(sid,split,key,other=False):
        return dict(id=sid,qid=str(sid),spectrum_id=str(sid),source='experimental',split=split,group=str(sid),
            precursor_mz=61.06479,collision_energy_nce=35,precursor_type='[M+H]+',
            peaks=[[29.03858,5 if other else 50],[31.01784,50 if other else 5],[45.03349,100]],
            targets=[[],[],[[key,1.]]],precursor_peak=[False]*3)
    records=[record(1,'train',by_smiles['CCO']['key']),record(2,'train',by_smiles['COC']['key'],True),
        record(3,'val',by_smiles['CCO']['key']),record(4,'val',by_smiles['COC']['key'],True),record(5,'val',unknown),
        record(6,'test',unknown)]
    index=[]
    with sqlite3.connect(directory/'records.sqlite') as db:
        db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY,source TEXT,split TEXT,qid TEXT,payload BLOB)')
        for r in records:
            db.execute('INSERT INTO records VALUES (?,?,?,?,?)',(r['id'],r['source'],r['split'],r['qid'],zlib.compress(json.dumps(r).encode())))
            index.append({k:r[k] for k in ['id','source','split','qid']}|dict(num_peaks=3,annotated_peaks=1))
    db.close()
    (directory/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    meta=dict(schema_version=1,label_semantics='unit test fixture',config=asdict(PrepareConfig()),reference_signature={},partial=True,
        hashes={name:digest(directory/name) for name in ['vocab.json','features.pt','index.jsonl','records.sqlite']})
    (directory/'meta.json').write_text(json.dumps(meta))
    return records


class PeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.records=fixture(self.root);self.catalog=Catalog(self.root)
        self.cfg=ModelConfig(hidden=32,heads=4,spectrum_layers=1,graph_layers=1,fourier_frequencies=4,dropout=0.)

    def tearDown(self):self.temp.cleanup()

    def test_hydrogen_difference_is_explicit_and_heavy_atom_mismatch_rejected(self):
        label=structure('CCO','C2H5O','[M+H]+')
        self.assertFalse(label['formula_smiles_consistent']);self.assertEqual(label['smiles_formula'],'C2H6O')
        with self.assertRaises(ValueError):structure('CCO','C3H5O','[M+H]+')

    def test_teacher_and_dag_are_not_student_inputs(self):
        r=self.records[0];poison={**r,'targets':[[['wrong',1.]]]*3,'dag_nodes':[{'secret':99}],'true_mol_metadata':{'smiles':'CCCC'}}
        a=spectrum_batch([r]);b=spectrum_batch([poison])
        for key in a:self.assertTrue(torch.equal(a[key],b[key]))
        self.assertEqual(read_spectrum(r),read_spectrum(poison))

    def test_set_encoder_is_peak_permutation_equivariant(self):
        torch.manual_seed(3);model=PeakAnnotator(self.cfg).eval();r=self.records[0]
        perm=[2,0,1];other={**r,'peaks':[r['peaks'][i] for i in perm]}
        with torch.no_grad():a=model.encode_spectrum(spectrum_batch([r]));b=model.encode_spectrum(spectrum_batch([other]))
        self.assertTrue(torch.allclose(a[:,perm],b,atol=1e-5))

    def test_mass_matched_negatives_and_bag_loss(self):
        records=self.records[:2];targets=training_targets(records,self.catalog,random.Random(0),32,64)
        self.assertEqual(int(targets['valid'].sum()),4)
        for _,_,row in zip(targets['positions'],targets['valid'],targets['indices']):
            for i in row:self.assertAlmostEqual(self.catalog.labels[targets['ids'][int(i)]]['mz'],45.03349)
        model=PeakAnnotator(self.cfg)
        q,f=model(spectrum_batch(records),self.catalog.graphs(targets['ids']))
        for mode in ['soft','set']:
            loss=retrieval_loss(q,f,targets,.07,mode);self.assertTrue(torch.isfinite(loss))
        all_positive={**targets,'positive':targets['valid']}
        self.assertAlmostEqual(float(retrieval_loss(q,f,all_positive,.07,'set').detach()),0.,places=6)

    def test_oov_is_not_silently_removed_from_accuracy_denominator(self):
        model=PeakAnnotator(self.cfg).eval();dataset=PeakDataset(self.root,['experimental'],'val')
        result=evaluate_model(model,dataset,self.catalog,torch.device('cpu'))
        self.assertEqual(result['teacher_annotated_peaks'],3);self.assertEqual(result['vocab_covered_peaks'],2)
        self.assertEqual(result['teacher_top3_all_annotated'],2/3)
        self.assertEqual(result['teacher_top3_vocab_covered'],1.)
        self.assertEqual(result['unannotated_fragment_peaks'],6)

    def test_loss_can_learn_context_to_distinguish_same_mass_structures(self):
        torch.manual_seed(5);model=PeakAnnotator(self.cfg);optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        records=self.records[:2];batch=spectrum_batch(records)
        targets=training_targets(records,self.catalog,random.Random(0));graphs=self.catalog.graphs(targets['ids'])
        values=[]
        for _ in range(70):
            optimizer.zero_grad();q,f=model(batch,graphs);loss=retrieval_loss(q,f,targets,.07,'soft')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();values.append(float(loss.detach()))
        self.assertLess(sum(values[-5:])/5,values[0]*.4)

    def test_sparse_nids_and_duplicate_states_form_one_label(self):
        entry=dict(identifiers={'id':'x','query_id':'1'},research_metadata={'candidate_index':0,'is_target':True},
            experiment_metadata={'precursor_mass':61.06479,'collision_energy_nce':35,'precursor_type':'[M+H]+'},
            predicted_ms2=[[45.03349,100]],experimental_ms2=[[45.03349,100]],fragment_matching=[0,0,0],
            dag_nodes=[{'nid':900,'SMILES':'CCCO','formula':'C3H9O','mass':61.06479,'in_nids':[],'final_prob':.1},
                {'nid':44,'SMILES':'CCO','formula':'C2H5O','mass':45.03349,'in_nids':[900],'final_prob':.2},
                {'nid':18,'SMILES':'OCC','formula':'C2H5O','mass':45.03349,'in_nids':[900],'final_prob':.3}])
        archive=self.root/'archive.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('visualizer/casmi_1.json',json.dumps([entry]))
        info=dict(id=1,qid='1',spectrum_id='x',candidate_index=0,source='predicted',split='train',group='g')
        rows,labels,audit=_query((str(archive),'1',[info],PrepareConfig()))
        record=json.loads(zlib.decompress(rows[0][4]));self.assertEqual(len(record['targets'][0]),1)
        self.assertEqual(record['targets'][0][0][1],1.);self.assertEqual(len(labels),1)
        info['split']='test';_,labels,_=_query((str(archive),'1',[info],PrepareConfig()))
        self.assertEqual(labels,{})  # Held-out fragments cannot enter the retrieval catalog.

    def test_signature_guard_rejects_wrong_catalog_or_resume_subset(self):
        from casmi_peaks.data import signature
        meta=json.loads((self.root/'meta.json').read_text());cfg={'model':asdict(self.cfg),'training':{'epochs':2}}
        ckpt=dict(task='peak_fragment_retrieval_v1',dataset_signature=signature(meta),model_config=cfg['model'],
            config=cfg,subset_signature='original',world_size=1)
        with self.assertRaises(ValueError):checkpoint_guard(ckpt,cfg,meta,'limited',resume=True)
        ckpt['dataset_signature']={}
        with self.assertRaises(ValueError):checkpoint_guard(ckpt,cfg,meta)

    def test_epoch_resume_restores_optimizer_schedule_and_dropout_rng(self):
        from casmi_peaks import engine
        cfg=dict(seed=7,sources=['experimental'],loss_mode='soft',model=asdict(self.cfg)|{'dropout':.1},
            training=dict(epochs=3,batch_size=2,eval_batch_size=3,learning_rate=.001,
                          max_target_peaks=8,negatives=4,cpu_threads=2,catalog_batch=8,workers=0))
        config=self.root/'config.json';config.write_text(json.dumps(cfg))
        full=self.root/'full';interrupted=self.root/'interrupted'
        original=engine.evaluate_model;calls=[]
        def interrupt(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('simulated interruption')
            return original(*args,**kwargs)
        with redirect_stdout(io.StringIO()):
            engine.train(config,self.root,full,'cpu')
            with patch.object(engine,'evaluate_model',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):engine.train(config,self.root,interrupted,'cpu')
            engine.train(config,self.root,interrupted,'cpu',resume=interrupted/'last.pt')
        a=torch.load(full/'last.pt',weights_only=True);b=torch.load(interrupted/'last.pt',weights_only=True)
        self.assertEqual(a['global_step'],b['global_step']);self.assertEqual(a['scheduler'],b['scheduler'])
        for key in a['model']:self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)

    def test_completed_stage_cannot_hide_limited_training_subset(self):
        from casmi_peaks.server import stage_state
        from casmi_peaks.data import signature
        meta=json.loads((self.root/'meta.json').read_text());cfg=dict(model=asdict(self.cfg),sources=['experimental'],training={'epochs':2})
        run=self.root/'stage';run.mkdir()
        payload=dict(dataset_signature=signature(meta),config=cfg,subset_signature='limited',epoch=1,bad_epochs=0)
        torch.save(payload,run/'last.pt');torch.save(payload,run/'best.pt')
        with self.assertRaisesRegex(ValueError,'subset'):stage_state(run,cfg,meta,self.root)


if __name__=='__main__':unittest.main()
