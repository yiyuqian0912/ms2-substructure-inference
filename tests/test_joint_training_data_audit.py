"""Regression checks for label-blind support and relation-only data use."""
from pathlib import Path
from contextlib import redirect_stdout
import io,json,random,sqlite3,tempfile,unittest,zlib
from unittest.mock import patch
import torch
from dataclasses import asdict
from test_peaks import fixture
from test_joint import relation_fixture
from casmi_peaks.data import Catalog,digest
from casmi_joint.relations import RelationStore,make_training
from casmi_joint.model import JointConfig
from casmi_joint.engine import train


class JointTrainingDataAudit(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.data=self.root/'data';self.records=fixture(self.data);self.catalog=Catalog(self.data)
        self.rel=self.root/'relations';relation_fixture(self.rel,self.data)
        self.store=RelationStore(self.rel,json.loads((self.data/'meta.json').read_text()))

    def tearDown(self):self.store.close();self.temp.cleanup()

    def test_support_candidates_do_not_follow_changed_teacher_positive(self):
        a=self.records[0];b={**a,'targets':[[],[],self.records[1]['targets'][2]]}
        def support(record,seed):
            t,_=make_training([record],self.catalog,self.store,random.Random(seed),8,1,4,'set',0)
            rows=t['positions'].tolist(); row=rows.index([0,2])
            return {t['ids'][i] for i in t['indices'][row,t['support_valid'][row]].tolist()},t,row
        omitted=False
        for seed in range(10):
            sa,t,row=support(a,seed);sb,_,_=support(b,seed)
            self.assertEqual(sa,sb)
            true={t['ids'][i] for i in t['indices'][row,t['positive'][row]].tolist()}
            omitted|=not bool(sa&true)
        self.assertTrue(omitted)

    def test_context_selection_backfills_after_empty_mass_pools(self):
        r={**self.records[0],'peaks':[[500.123,1000.],[29.03858,500.],[45.03349,100.]],'targets':[[],[],self.records[0]['targets'][2]]}
        t,_=make_training([r],self.catalog,self.store,random.Random(1),8,4,4,'set',1)
        contexts=t['positions'][~t['supervised']].tolist()
        self.assertEqual(contexts,[[0,1]])

    def test_unannotated_experimental_record_receives_relation_only_replay(self):
        self.store.close()
        r={**self.records[1],'targets':[[],[],[]]}
        with sqlite3.connect(self.data/'records.sqlite') as db:
            db.execute('UPDATE records SET payload=? WHERE id=?',(zlib.compress(json.dumps(r).encode()),r['id']))
        index=[json.loads(s) for s in (self.data/'index.jsonl').read_text().splitlines()]
        for item in index:
            if item['id']==r['id']:item['annotated_peaks']=0
        (self.data/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index))
        meta=json.loads((self.data/'meta.json').read_text())
        for name in ('index.jsonl','records.sqlite'):meta['hashes'][name]=digest(self.data/name)
        (self.data/'meta.json').write_text(json.dumps(meta))
        import shutil
        shutil.rmtree(self.rel);relation_fixture(self.rel,self.data)
        cfg=dict(seed=7,sources=['experimental'],loss_mode='set',model=asdict(JointConfig(hidden=16,heads=4,spectrum_layers=1,graph_layers=1,mass_frequencies=4,relation_rank=4,dropout=0.)),
            joint=dict(candidate_k=16,strength=0.,iterations=1,neighbors=2,support_k=2,damping=.5),
            training=dict(epochs=1,batch_size=1,learning_rate=.001,max_target_peaks=8,negatives=4,relation_pairs=4,relation_weight=.1,catalog_batch=4,cpu_threads=2))
        path=self.root/'cfg.json';path.write_text(json.dumps(cfg))
        with redirect_stdout(io.StringIO()):train(path,self.data,self.rel,self.root/'run','cpu')
        checkpoint=torch.load(self.root/'run/last.pt',weights_only=True)
        self.assertEqual(checkpoint['relation_only_ids'],[r['id']])
        self.assertEqual(checkpoint['limited_training'],False)


if __name__=='__main__':unittest.main()
