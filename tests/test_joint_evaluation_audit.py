"""Observation-only candidate encoding and calibration scientific contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_peaks import fixture
from casmi_peaks.data import Catalog,digest
from casmi_peaks.engine import encode_catalog
from casmi_joint.model import JointAnnotator,JointConfig
from casmi_joint.predict import (CatalogSubsetEmbeddings,encode_observed_catalog,observed_pools,
                                 prepare_record,rank_prepared,rank_record)
from casmi_joint.evaluate import evaluate_model,load_calibration


class EvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(23)
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.records=fixture(self.root); self.catalog=Catalog(self.root)
        self.cfg=JointConfig(hidden=32,heads=4,spectrum_layers=1,graph_layers=2,mass_frequencies=8,relation_rank=8,dropout=0.)
        self.model=JointAnnotator(self.cfg).eval(); self.device=torch.device('cpu')
        self.settings=dict(candidate_k=16,strength=2.,iterations=2,neighbors=3,support_k=4,damping=.5)
        self.record=deepcopy(self.records[0])
        for key in ('peaks','targets','precursor_peak'): self.record[key]=self.record[key][1:]

    def tearDown(self): self.temp.cleanup()

    def test_subset_keeps_unannotated_support_candidates_and_ignores_targets(self):
        subset=encode_observed_catalog(self.model,self.catalog,[self.record],self.device,4)
        expected={k for pool in observed_pools(self.record,self.catalog) for k in pool}
        self.assertEqual(set(subset.ids.tolist()),expected)
        self.assertLess(len(subset.ids),len(self.catalog.labels))
        support=self.catalog.compatible(self.record['peaks'][0][0],self.record['precursor_type'])
        self.assertTrue(support); self.assertTrue(set(support)<=expected)
        self.assertFalse(self.record['targets'][0])
        poison={**self.record,'targets':[[['secret',1.]]]*2,'dag_nodes':[{'forbidden':True}]}
        other=encode_observed_catalog(self.model,self.catalog,[poison],self.device,4)
        self.assertTrue(np.array_equal(subset.ids,other.ids)); self.assertTrue(np.array_equal(subset.values,other.values))
        missing=next(k for k in range(len(self.catalog.labels)) if k not in expected)
        with self.assertRaisesRegex(ValueError,'not encoded'): subset[[missing]]

    def test_full_and_subset_embeddings_preserve_unary_joint_ranking_and_scores(self):
        full=encode_catalog(self.model,self.catalog,self.device,4)
        subset=encode_observed_catalog(self.model,self.catalog,[self.record],self.device,4)
        a=rank_record(self.model,self.record,self.catalog,self.device,self.settings,full)
        b=rank_record(self.model,self.record,self.catalog,self.device,self.settings,subset)
        self.assertEqual(a['pools'],b['pools']); self.assertEqual(a['unary_rank'],b['unary_rank']); self.assertEqual(a['joint_rank'],b['joint_rank'])
        for key in ('unary_scores','joint_scores'):
            for x,y in zip(a[key],b[key]):
                self.assertEqual(set(x),set(y))
                np.testing.assert_allclose(list(x.values()),list(y.values()),atol=1e-5,rtol=1e-5)

    def test_unary_preparation_reused_for_strength_grid_matches_individual_calls(self):
        full=encode_catalog(self.model,self.catalog,self.device,4)
        prepared=prepare_record(self.model,self.record,self.catalog,self.device,self.settings,full)
        for strength in (0.,.5,2.):
            settings={**self.settings,'strength':strength}
            shared=rank_prepared(self.model,prepared,settings)
            individual=rank_record(self.model,self.record,self.catalog,self.device,settings,full)
            self.assertEqual(shared,individual)
        with self.assertRaisesRegex(ValueError,'candidate_k'):rank_prepared(self.model,prepared,{**self.settings,'candidate_k':32})

    def test_evaluator_prepares_once_per_spectrum_and_full_subset_metrics_match(self):
        from casmi_joint import evaluate as module
        records=[self.record]
        original=module.prepare_record
        with patch.object(module,'prepare_record',wraps=original) as mock:
            subset_result=evaluate_model(self.model,records,self.catalog,self.device,self.settings,strengths=[0.,.5,2.],catalog_batch=4)
            self.assertEqual(mock.call_count,len(records))
        full=encode_catalog(self.model,self.catalog,self.device,4)
        full_view=CatalogSubsetEmbeddings(list(range(len(full))),full,len(full))
        with patch.object(module,'encode_observed_catalog',return_value=full_view):
            full_result=evaluate_model(self.model,records,self.catalog,self.device,self.settings,strengths=[0.,.5,2.],catalog_batch=4)
        subset_result.pop('evaluation_encoding');full_result.pop('evaluation_encoding')
        self.assertEqual(subset_result,full_result)

    def test_calibration_binds_checkpoint_dataset_grid_and_finite_strength(self):
        checkpoint=self.root/'checkpoint.pt'; sig={'catalog':'signed-test-catalog'}
        torch.save({'dataset_signature':sig},checkpoint)
        packet=dict(calibration_schema_version=2,split='val',checkpoint_sha256=digest(checkpoint),
            dataset_signature=sig,base_settings=self.settings,strength_grid=[0.,2.],selected_strength=2.,
            metrics={'joint_by_strength':{'0':{},'2':{}}})
        path=self.root/'calibration.json'
        def write(value): path.write_text(json.dumps(value))
        write(packet); self.assertEqual(load_calibration(path,checkpoint,self.settings)['strength'],2.)
        for value in (-1.,float('nan'),float('inf'),1.,True):
            write({**packet,'selected_strength':value})
            with self.assertRaises(ValueError):load_calibration(path,checkpoint,self.settings,sig)
        write({**packet,'dataset_signature':{'catalog':'wrong'}})
        with self.assertRaisesRegex(ValueError,'signature'):load_calibration(path,checkpoint,self.settings,sig)
        write({**packet,'strength_grid':[0.,float('nan')]})
        with self.assertRaisesRegex(ValueError,'finite'):load_calibration(path,checkpoint,self.settings,sig)
        write({**packet,'calibration_schema_version':1})
        with self.assertRaisesRegex(ValueError,'calibration'):load_calibration(path,checkpoint,self.settings,sig)


if __name__=='__main__':unittest.main()
