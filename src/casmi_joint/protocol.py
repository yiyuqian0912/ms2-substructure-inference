"""Index-only query/scaffold intersection reference; never reads spectrum/DAG data.

The original CASMI candidate-scaffold split is retained. A row is usable only
when its original split agrees with its query's experimental target split.
Thus whole queries are disjoint without moving a held-out candidate scaffold
into training. Rejected prediction rows are signed and quarantined, not relabelled.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


PROTOCOL = 'query_scaffold_intersection_v1'
_SIGNATURE_FIELDS = ('schema_version','index_sha256','archive_sha256','config','fp_bits','node_dim','edge_dim')


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _json_bytes(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()


def validate_query_reference(directory,source=None):
    """Validate all signed sidecar files and, optionally, the unchanged source."""
    directory=Path(directory)
    meta=json.loads((directory/'meta.json').read_text())
    if meta.get('protocol')!=PROTOCOL or not meta.get('index_only'):
        raise ValueError('Not a query/scaffold intersection reference')
    for name,expected in meta['hashes'].items():
        if _digest(directory/name)!=expected:
            raise ValueError(f'Query protocol checksum mismatch: {name}')
    if meta['index_sha256']!=meta['hashes']['index.jsonl']:
        raise ValueError('Query protocol index signature mismatch')
    if meta.get('prepare_code_sha256')!=_digest(__file__):
        raise ValueError('Query protocol implementation changed; use a new output directory')
    if source is not None:
        source=Path(source)
        if (_digest(source/'index.jsonl')!=meta['source_reference']['index_sha256'] or
                _digest(source/'meta.json')!=meta['source_reference']['meta_sha256']):
            raise ValueError('Query protocol source reference changed')
    return meta


def prepare_query_reference(reference,output):
    """Build a new reference compatible with casmi_peaks.data.prepare.

    This is deliberately not a casmi_neural prepared SQLite dataset. Call the
    legacy ensure_prepared on `reference`, then this helper, then prepare the
    peak cache from the returned output directory.
    """
    reference,output=Path(reference),Path(output)
    if reference.resolve()==output.resolve():
        raise ValueError('Query protocol output must differ from the legacy reference')
    meta=json.loads((reference/'meta.json').read_text())
    if meta.get('schema_version')!=1 or meta.get('config',{}).get('split')!='scaffold':
        raise ValueError('Query protocol requires an original scaffold reference')
    data=(reference/'index.jsonl').read_bytes()
    source_hash=hashlib.sha256(data).hexdigest()
    if source_hash!=meta['index_sha256']:
        raise ValueError('Source reference index checksum mismatch')
    rows=[json.loads(line) for line in data.splitlines()]
    if len(rows)!=meta['deduplicated_records'] or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Source reference record count/IDs disagree')
    if not rows:
        raise ValueError('Empty source reference')
    targets=defaultdict(list); scaffold_splits=defaultdict(set)
    for row in rows:
        if row['split'] not in ('train','val','test') or row['source'] not in ('predicted','experimental'):
            raise ValueError('Unsupported source/split in reference')
        scaffold_splits[row['group']].add(row['split'])
        if row['source']=='experimental':
            targets[str(row['qid'])].append(row)
    # Check the true-precursor grouping explicitly before touching candidates.
    query_split={}; true_scaffolds=defaultdict(set)
    for qid,records in targets.items():
        splits={r['split'] for r in records}; groups={r['group'] for r in records}
        if len(splits)!=1 or len(groups)!=1:
            raise ValueError(f'Experimental target query {qid} has inconsistent split/scaffold')
        query_split[qid]=next(iter(splits)); true_scaffolds[next(iter(groups))].update(splits)
    if any(len(splits)!=1 for splits in true_scaffolds.values()):
        raise ValueError('True precursor scaffolds cross experimental query splits')
    if any(len(splits)!=1 for splits in scaffold_splits.values()):
        raise ValueError('Candidate scaffold groups cross source splits')
    if {str(r['qid']) for r in rows}!=set(query_split):
        raise ValueError('Each query must have experimental target records')
    if len(query_split)!=meta.get('queries'):
        raise ValueError('Source reference query count differs')
    actual_counts=dict(Counter(f"{r['source']}/{r['split']}" for r in rows))
    if meta.get('counts')!=actual_counts:
        raise ValueError('Source reference source/split counts differ')
    kept=[]; quarantined=[]
    for row in sorted(rows,key=lambda r:r['id']):
        expected=query_split[str(row['qid'])]
        if row['split']==expected:
            kept.append(row)
        else:
            if row['source']=='experimental':
                raise ValueError('Experimental target must never be quarantined')
            quarantined.append({**row,'query_split':expected,'exclusion_reason':'candidate_scaffold_split_differs_from_query_target_split'})
    query_sets=defaultdict(set); kept_groups=defaultdict(set)
    for row in kept:
        query_sets[str(row['qid'])].add(row['split']); kept_groups[row['group']].add(row['split'])
    if any(len(v)!=1 for v in query_sets.values()) or any(len(v)!=1 for v in kept_groups.values()):
        raise ValueError('Intersection failed query/scaffold isolation')
    source_complete=(meta.get('queries')==246 and len(rows)==36279 and not meta.get('partial',False))
    counts=dict(Counter(f"{r['source']}/{r['split']}" for r in kept))
    # Completeness means applying the signed rule to the entire source, not
    # preserving predictions that would violate the held-out query protocol.
    config={**meta['config'],'split':PROTOCOL,'protocol':PROTOCOL}
    payloads={
        'index.jsonl':b''.join(_json_bytes(r) for r in kept),
        'quarantine.jsonl':b''.join(_json_bytes(r) for r in quarantined),
        'query_splits.json':_json_bytes(query_split),
    }
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}
    result={
        **{k:meta[k] for k in ('schema_version','archive_sha256','fp_bits','node_dim','edge_dim')},
        'config':config,'index_sha256':hashes['index.jsonl'],'queries':len(query_split),
        'raw_records':len(kept),'deduplicated_records':len(kept),'expected_records':len(kept),
        'counts':counts,'group_counts':dict(Counter(next(iter(s)) for s in kept_groups.values())),
        'query_counts':dict(Counter(query_split.values())),
        'protocol':PROTOCOL,'index_only':True,'source_complete':source_complete,
        'protocol_complete':source_complete,'partial':not source_complete,
        'excluded_records':len(quarantined),'excluded_counts':dict(Counter(f"{r['source']}/{r['split']}" for r in quarantined)),
        'experimental_records_retained':sum(r['source']=='experimental' for r in kept),
        'isolation':{'query_disjoint':True,'candidate_scaffold_disjoint':True,'true_precursor_scaffold_disjoint':True},
        'source_reference':{**{k:meta[k] for k in _SIGNATURE_FIELDS},'meta_sha256':_digest(reference/'meta.json'),
                            'records':len(rows),'queries':meta['queries'],'raw_records':meta.get('raw_records')},
        'hashes':hashes,'prepare_code_sha256':_digest(__file__),
        'semantics':'Keep source split == experimental target query split. Original scaffold groups and labels are never moved. Quarantined predictions are not used for vocabulary or training.',
    }
    if output.exists() and any(output.iterdir()):
        existing=validate_query_reference(output,reference)
        if existing!=result:
            raise ValueError('Existing query protocol output differs; choose a new directory')
        return existing
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=output.name+'.building-',dir=output.parent))
    try:
        for name,value in payloads.items():
            (temporary/name).write_bytes(value)
        (temporary/'meta.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    print(json.dumps(prepare_query_reference(args.reference,args.output),indent=2),flush=True)
