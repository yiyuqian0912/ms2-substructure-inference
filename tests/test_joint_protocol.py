"""Query/scaffold intersection reference: isolation and signed provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from casmi_joint.protocol import prepare_query_reference,validate_query_reference,PROTOCOL


def source_fixture(path,rows=None):
    path.mkdir()
    if rows is None:
        rows=[
            dict(id=1,qid='1',source='experimental',split='train',group='target-A'),
            dict(id=2,qid='1',source='predicted',split='train',group='candidate-A'),
            dict(id=3,qid='1',source='predicted',split='test',group='target-C'),
            dict(id=4,qid='2',source='experimental',split='val',group='target-B'),
            dict(id=5,qid='2',source='predicted',split='train',group='candidate-A'),
            dict(id=6,qid='2',source='predicted',split='val',group='candidate-B'),
            dict(id=7,qid='3',source='experimental',split='test',group='target-C'),
            dict(id=8,qid='3',source='predicted',split='train',group='candidate-D'),
            dict(id=9,qid='3',source='predicted',split='test',group='candidate-C'),
        ]
    rows=[{**r,'spectrum_id':f"{r['qid']}_35",'candidate_index':r['id'],'num_peaks':3,'num_nodes':2,'num_edges':1} for r in rows]
    payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
    (path/'index.jsonl').write_bytes(payload)
    meta=dict(schema_version=1,config=dict(split='scaffold',split_seed=7,tolerance_da=.007,tolerance_ppm=5.,intensity_power=.5),
        archive_sha256='not-a-real-archive-no-DAG-reading-needed',index_sha256=hashlib.sha256(payload).hexdigest(),
        fp_bits=2048,node_dim=22,edge_dim=15,queries=len({r['qid'] for r in rows}),raw_records=len(rows),deduplicated_records=len(rows),
        counts=dict(Counter(f"{r['source']}/{r['split']}" for r in rows)))
    (path/'meta.json').write_text(json.dumps(meta))
    return rows,meta


class QueryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.output=self.root/'intersection'

    def tearDown(self):self.temp.cleanup()

    def test_intersection_preserves_both_query_and_scaffold_disjointness(self):
        rows,source_meta=source_fixture(self.source)
        meta=prepare_query_reference(self.source,self.output)
        kept=[json.loads(s) for s in (self.output/'index.jsonl').read_text().splitlines()]
        self.assertEqual([r['id'] for r in kept],[1,2,4,6,7,9])
        self.assertEqual(meta['excluded_records'],3);self.assertEqual(meta['experimental_records_retained'],3)
        self.assertEqual(meta['query_counts'],dict(train=1,val=1,test=1))
        self.assertTrue(all(meta['isolation'].values()));self.assertFalse(meta['source_complete'])
        self.assertEqual(meta['protocol'],PROTOCOL);self.assertTrue(meta['index_only'])
        original={r['id']:r for r in rows}
        for r in kept:
            self.assertEqual(r,original[r['id']])  # No split, scaffold or identity relabelling.
        discarded=[json.loads(s) for s in (self.output/'quarantine.jsonl').read_text().splitlines()]
        self.assertEqual({r['id'] for r in discarded},{3,5,8})
        self.assertEqual(meta['source_reference']['index_sha256'],source_meta['index_sha256'])
        self.assertEqual(validate_query_reference(self.output,self.source),meta)

    def test_idempotent_reuse_checks_all_sidecar_hashes(self):
        source_fixture(self.source); first=prepare_query_reference(self.source,self.output)
        self.assertEqual(prepare_query_reference(self.source,self.output),first)
        with (self.output/'quarantine.jsonl').open('a') as stream:stream.write('{}\n')
        with self.assertRaisesRegex(ValueError,'checksum'):prepare_query_reference(self.source,self.output)

    def test_source_checksum_and_record_count_are_verified(self):
        _,meta=source_fixture(self.source)
        with (self.source/'index.jsonl').open('a') as stream:stream.write('{}\n')
        with self.assertRaisesRegex(ValueError,'checksum'):prepare_query_reference(self.source,self.output)

    def test_target_query_cannot_have_multiple_splits(self):
        rows=[dict(id=1,qid='1',source='experimental',split='train',group='A'),
              dict(id=2,qid='1',source='experimental',split='test',group='B')]
        source_fixture(self.source,rows)
        with self.assertRaisesRegex(ValueError,'inconsistent'):prepare_query_reference(self.source,self.output)

    def test_true_precursor_scaffold_must_be_isolated(self):
        rows=[dict(id=1,qid='1',source='experimental',split='train',group='A'),
              dict(id=2,qid='2',source='experimental',split='test',group='A')]
        source_fixture(self.source,rows)
        with self.assertRaisesRegex(ValueError,'True precursor scaffolds'):prepare_query_reference(self.source,self.output)

    def test_candidate_scaffold_must_also_be_isolated(self):
        rows=[dict(id=1,qid='1',source='experimental',split='train',group='A'),
              dict(id=2,qid='2',source='experimental',split='test',group='B'),
              dict(id=3,qid='1',source='predicted',split='train',group='C'),
              dict(id=4,qid='2',source='predicted',split='test',group='C')]
        source_fixture(self.source,rows)
        with self.assertRaisesRegex(ValueError,'Candidate scaffold'):prepare_query_reference(self.source,self.output)

    def test_missing_experimental_query_is_rejected(self):
        rows=[dict(id=1,qid='1',source='experimental',split='train',group='A'),
              dict(id=2,qid='2',source='predicted',split='train',group='B')]
        source_fixture(self.source,rows)
        with self.assertRaisesRegex(ValueError,'Each query'):prepare_query_reference(self.source,self.output)


if __name__=='__main__':unittest.main()
