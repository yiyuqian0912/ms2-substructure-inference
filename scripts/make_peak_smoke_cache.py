#!/usr/bin/env python3
"""Build a clearly marked, train-only reduced vocabulary for local execution checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import zlib
import torch
from casmi_peaks.chem import graph_features
from casmi_peaks.data import Catalog,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vocab-size',type=int,default=512);p.add_argument('--per-split',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Choose a new smoke cache directory')
    a.output.mkdir(parents=True)
    cat=Catalog(a.data,load_features=False)
    labels=sorted(sorted(cat.labels,key=lambda v:(-v['train_peak_occurrences'],v['key']))[:a.vocab_size],key=lambda v:v['key'])
    labels=[{**v,'id':i} for i,v in enumerate(labels)]
    keys={v['key'] for v in labels};index=[];counts=Counter();db=sqlite3.connect((a.data/'records.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    with sqlite3.connect(a.output/'records.sqlite') as target:
        target.execute('CREATE TABLE records (id INTEGER PRIMARY KEY,source TEXT,split TEXT,qid TEXT,payload BLOB)')
        for line in (a.data/'index.jsonl').read_text().splitlines():
            item=json.loads(line);group=item['source']+'/'+item['split']
            if counts[group]>=a.per_split:continue
            row=db.execute('SELECT * FROM records WHERE id=?',(item['id'],)).fetchone()
            record=json.loads(zlib.decompress(row[-1]))
            if item['split']=='train' and not any(any(k in keys for k,_ in t) for t in record['targets']):continue
            target.execute('INSERT INTO records VALUES (?,?,?,?,?)',row);index.append(item);counts[group]+=1
    db.close()
    (a.output/'index.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in index))
    (a.output/'vocab.json').write_text(json.dumps(labels,separators=(',',':'))+'\n')
    atoms=[];edges=[];bonds=[];numeric=[];npointer=[0];epointer=[0]
    for label in labels:
        n,e,b,v=graph_features(label);atoms.append(n);edges.append(e);bonds.append(b);numeric.append(v)
        npointer.append(npointer[-1]+len(n));epointer.append(epointer[-1]+e.shape[1])
    torch.save(dict(atoms=torch.cat(atoms),edges=torch.cat(edges,1),bonds=torch.cat(bonds),numeric=torch.stack(numeric),
        node_ptr=torch.tensor(npointer),edge_ptr=torch.tensor(epointer)),a.output/'features.pt')
    meta=json.loads((a.data/'meta.json').read_text());meta.update(partial=True,vocabulary=len(labels),counts=dict(counts),
        smoke_note='Execution checks only: frequency-selected training labels and a small spectrum subset; not a benchmark result.',
        hashes={name:digest(a.output/name) for name in ['vocab.json','features.pt','index.jsonl','records.sqlite']})
    (a.output/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),counts=dict(counts),vocabulary=len(labels))),flush=True)


if __name__=='__main__':main()
