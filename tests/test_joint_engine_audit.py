"""Accumulation, initialization lineage and deployment orchestration regressions."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib
import torch
from test_peaks import fixture
from test_joint import relation_fixture
from casmi_peaks.data import digest
from casmi_joint.model import JointConfig
from casmi_joint.engine import train,read_config
from casmi_joint import server


class EngineAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.data=self.root/'data'; records=fixture(self.data)
        labels=json.loads((self.data/'vocab.json').read_text()); keys={v['smiles']:v['key'] for v in labels}
        # Three spectra provide 1, 3 and 2 supervised peaks. The 29/31 peaks
        # have singleton candidate pools, still count in the set-loss mean.
        records[1]['targets'][0]=[[keys['CC'],1.]]; records[1]['targets'][1]=[[keys['CO'],1.]]
        third=deepcopy(records[0]);third.update(id=7,qid='7',spectrum_id='7',group='7')
        third['targets'][0]=[[keys['CC'],1.]]; records.append(third)
        index=[]
        with sqlite3.connect(self.data/'records.sqlite') as db:
            for record in records:
                db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?)',(record['id'],record['source'],record['split'],record['qid'],zlib.compress(json.dumps(record).encode())))
                index.append({k:record[k] for k in ('id','source','split','qid')}|dict(num_peaks=len(record['peaks']),annotated_peaks=sum(bool(x) for x in record['targets'])))
        db.close()
        (self.data/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index))
        meta=json.loads((self.data/'meta.json').read_text());meta['partial']=False
        meta['hashes']={name:digest(self.data/name) for name in meta['hashes']}
        (self.data/'meta.json').write_text(json.dumps(meta)); self.meta=meta
        self.relations=self.root/'relations';self.rmeta=relation_fixture(self.relations,self.data)
        self.cfg=dict(seed=7,sources=['experimental'],loss_mode='set',
            model=asdict(JointConfig(hidden=16,heads=4,spectrum_layers=1,graph_layers=1,mass_frequencies=4,relation_rank=4,dropout=0.)),
            joint=dict(candidate_k=16,strength=0.,iterations=2,neighbors=3,support_k=4,damping=.5),
            training=dict(epochs=1,batch_size=2,eval_batch_size=3,accumulation=1,learning_rate=.001,weight_decay=0.,
                gradient_clip=1e6,max_target_peaks=8,context_peaks=0,negatives=4,relation_pairs=4,relation_weight=0.,
                unary_weight=0.,cpu_threads=2,catalog_batch=4,workers=0))

    def tearDown(self):self.temp.cleanup()

    def config(self,name,**training):
        config=deepcopy(self.cfg); config['training'].update(training)
        path=self.root/(name+'.json');path.write_text(json.dumps(config));return path

    def test_accumulation_matches_large_batch_for_unequal_counts_and_partial_window(self):
        original=torch.optim.AdamW
        def run(name,batch,accumulation):
            snapshots=[]
            class RecordingAdamW(original):
                def step(self,*args,**kwargs):
                    snapshots.append([None if p.grad is None else p.grad.detach().clone() for group in self.param_groups for p in group['params']])
                    return super().step(*args,**kwargs)
            with patch.object(torch.optim,'AdamW',RecordingAdamW),redirect_stdout(io.StringIO()):
                train(self.config(name,batch_size=batch,accumulation=accumulation),self.data,self.relations,self.root/name,'cpu')
            ckpt=torch.load(self.root/name/'last.pt',weights_only=True)
            return snapshots,ckpt
        large,a=run('large',2,1);micro,b=run('micro',1,2)
        # Three records produce a two-record window and a final one-record window.
        self.assertEqual(len(large),2);self.assertEqual(len(micro),2)
        self.assertEqual(a['global_step'],b['global_step']);self.assertEqual(a['scheduler'],b['scheduler'])
        errors=[]
        for large_step,micro_step in zip(large,micro):
            for x,y in zip(large_step,micro_step):
                self.assertEqual(x is None,y is None)
                if x is not None:
                    errors.append(float((x-y).abs().max()))
                    self.assertTrue(torch.allclose(x,y,atol=3e-5,rtol=3e-4),f'gradient mismatch {errors[-1]}')
        self.assertGreater(len(errors),0)

    def test_limited_initialization_sha_and_partial_status_propagate(self):
        config=self.config('lineage')
        with redirect_stdout(io.StringIO()):
            train(config,self.data,self.relations,self.root/'limited','cpu',limit=1)
            source=self.root/'limited/best.pt'
            train(config,self.data,self.relations,self.root/'child','cpu',initialize=source)
            child=self.root/'child/best.pt'
            train(config,self.data,self.relations,self.root/'grandchild','cpu',initialize=child)
        first=torch.load(source,weights_only=True); second=torch.load(child,weights_only=True)
        third=torch.load(self.root/'grandchild/last.pt',weights_only=True)
        self.assertFalse(first['dataset_signature']['partial'])
        self.assertTrue(first['limited_training']);self.assertTrue(second['limited_training']);self.assertTrue(third['limited_training'])
        self.assertEqual(second['initialization']['checkpoint_sha256'],digest(source))
        self.assertTrue(second['initialization']['partial']);self.assertTrue(third['initialization']['partial'])
        self.assertEqual(third['initialization']['checkpoint_sha256'],digest(child))
        with self.assertRaisesRegex(ValueError,'partial|subset'):
            server.stage_state(self.root/'limited',read_config(config),self.meta,self.rmeta,self.data)

    def test_production_finetune_rejects_unfinished_pretrain_before_launch(self):
        config=self.config('production');runroot=self.root/'production'
        (runroot/'pretrain').mkdir(parents=True);(runroot/'pretrain/best.pt').write_bytes(b'exists')
        args=SimpleNamespace(pretrain_config=config,finetune_config=config,run_root=runroot,cpu=True,data=self.data,
                             relations=self.relations,gpus=1,port=29512)
        with patch.object(server,'stage_state',side_effect=['new','resume']),patch.object(server,'run') as launch:
            with self.assertRaisesRegex(ValueError,'completed full-data'):server.launch('finetune',args,self.meta,self.rmeta)
            launch.assert_not_called()
        with patch.object(server,'stage_state',side_effect=['new','complete']),patch.object(server,'run') as launch:
            server.launch('finetune',args,self.meta,self.rmeta)
            command=launch.call_args.args[0]
            self.assertIn('--initialize',command); self.assertIn(runroot/'pretrain/best.pt',command)

    def test_explicit_checkpoint_preserves_auto_or_explicit_calibration(self):
        from casmi_joint import evaluate,infer
        runroot=self.root/'deploy';runroot.mkdir()
        calibration=runroot/'calibration.json';calibration.write_text('{}')
        override=self.root/'override.json';override.write_text('{}')
        checkpoint=self.root/'explicit.pt';checkpoint.write_bytes(b'not-read-by-mock')
        spectrum=self.root/'spectrum.json';spectrum.write_text('{}')
        for command,module,name,position in [('evaluate',evaluate,'evaluate',5),('infer',infer,'infer',6)]:
            for explicit in (False,True):
                with self.subTest(command=command,explicit_calibration=explicit):
                    argv=['peak-aime-v2',command,'--cpu','--protocol','legacy','--run-root',str(runroot),
                          '--data',str(self.data),'--checkpoint',str(checkpoint)]
                    if command=='infer':argv+=['--input',str(spectrum)]
                    if explicit:argv+=['--calibration',str(override)]
                    with patch.object(sys,'argv',argv),patch.object(module,name) as called:
                        server.main()
                    self.assertEqual(called.call_args.args[0],checkpoint)
                    self.assertEqual(called.call_args.args[position],override if explicit else calibration)


if __name__=='__main__':unittest.main()
