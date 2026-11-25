"""Versioned supervision sidecar. Test DAGs are never used or exported."""
from collections import defaultdict, Counter, deque, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import zipfile
import zlib
import torch
from casmi_peaks.chem import structure
from casmi_peaks.data import Catalog,digest,signature

TYPES=('direct','reachable','siblings','common_precursor')


def graph_record(entry,keys):
    nodes=entry['dag_nodes']; by_nid={n['nid']:i for i,n in enumerate(nodes)}
    if len(by_nid)!=len(nodes): raise ValueError('Duplicate DAG nid')
    children=[]; labels=[]; probs=[]
    ion=entry['experiment_metadata']['precursor_type']
    for node in nodes:
        row=[]
        for child in node['out_nids']:
            if child not in by_nid: raise ValueError('Dangling DAG child')
            row.append(by_nid[child])
        children.append(sorted(set(row)))
        for parent in node['in_nids']:
            if parent not in by_nid or node['nid'] not in nodes[by_nid[parent]]['out_nids']:
                raise ValueError('Inconsistent DAG parents/children')
        for child in row:
            if node['nid'] not in nodes[child]['in_nids']: raise ValueError('Inconsistent DAG children/parents')
        key=None
        if node['in_nids']:
            try: key=structure(node['SMILES'],node['formula'],ion)['key']
            except ValueError: pass
        labels.append(keys.get(key,-1)); p=float(node.get('final_prob',0))
        if not math.isfinite(p) or p<0: raise ValueError('Invalid DAG contribution')
        probs.append(p)
    result=dict(nids=[n['nid'] for n in nodes],labels=labels,children=children,probabilities=probs)
    GraphView(result)  # Validate acyclicity before persisting.
    return result


class GraphView:
    def __init__(self,record):
        self.record=record; children=record['children']; n=len(children)
        parents=[set() for _ in range(n)]; degree=[0]*n
        for i,row in enumerate(children):
            for j in row:
                if not 0<=j<n: raise ValueError('Invalid DAG child position')
                parents[j].add(i); degree[j]+=1
        todo=deque(i for i,d in enumerate(degree) if d==0); order=[]
        while todo:
            i=todo.popleft(); order.append(i)
            for j in children[i]:
                degree[j]-=1
                if degree[j]==0: todo.append(j)
        if len(order)!=n: raise ValueError('Cycle in fragmentation DAG')
        self.descendants=[0]*n
        for i in reversed(order):
            for j in children[i]: self.descendants[i]|=(1<<j)|self.descendants[j]
        self.parents=parents; self.positions=defaultdict(list)
        for i,k in enumerate(record['labels']):
            if k>=0: self.positions[k].append(i)
        self.keys=set(self.positions)
        self.edges=sorted({(record['labels'][i],record['labels'][j]) for i,row in enumerate(children) for j in row
                    if record['labels'][i]>=0 and record['labels'][j]>=0 and record['labels'][i]!=record['labels'][j]})
        self.siblings=[]
        for row in children:
            valid=[j for j in row if record['labels'][j]>=0]
            if len(valid)>1: self.siblings.append(valid)

    def flags(self,a,b):
        # Existential aggregation across state occurrences avoids marking a
        # label pair negative merely because another occurrence had no edge.
        direct=reachable=sibling=False
        for i in self.positions[a]:
            for j in self.positions[b]:
                direct |= j in self.record['children'][i]
                reachable |= bool(self.descendants[i] & (1<<j))
                sibling |= bool(self.parents[i] & self.parents[j])
        return [float(direct),float(reachable),float(sibling),1.]

    def sample(self,rng,count):
        keys=sorted(self.keys)
        if len(keys)<2: return []
        output=[]; labels=self.record['labels']
        for k in range(count):
            if k%4==0 and self.edges: a,b=rng.choice(self.edges)
            elif k%4==1:
                a=rng.choice(keys); i=rng.choice(self.positions[a])
                pool=sorted({labels[j] for j in range(len(labels)) if labels[j]>=0 and labels[j]!=a and self.descendants[i]&(1<<j)})
                a,b=(a,rng.choice(pool)) if pool else rng.sample(keys,2)
            elif k%4==2 and self.siblings:
                row=rng.choice(self.siblings); i,j=rng.sample(row,2); a,b=labels[i],labels[j]
                if a==b: a,b=rng.sample(keys,2)
            else: a,b=rng.sample(keys,2)
            output.append((a,b,self.flags(a,b)))
        return output


def prepare_relations(archive,reference,data,output):
    """Only train/val entries selected by the signed peak index are exported."""
    archive,reference,data,output=map(Path,(archive,reference,data,output))
    meta=json.loads((data/'meta.json').read_text())
    if digest(archive)!=meta['reference_signature']['archive_sha256']: raise ValueError('Archive identity differs')
    if digest(reference/'index.jsonl')!=meta['reference_signature']['index_sha256']: raise ValueError('Reference identity differs')
    for name in ('index.jsonl','vocab.json'):
        if digest(data/name)!=meta['hashes'][name]: raise ValueError('Peak cache checksum differs')
    selected={r['id']:r for r in map(json.loads,(data/'index.jsonl').read_text().splitlines()) if r['split'] in ('train','val')}
    grouped=defaultdict(list)
    for line in (reference/'index.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['id'] not in selected: continue
        p=selected[row['id']]
        if any(p[k]!=row[k] for k in ('source','split','qid')): raise ValueError('Reference/peak split disagreement')
        grouped[row['qid']].append(row)
    if sum(map(len,grouped.values()))!=len(selected): raise ValueError('Missing reference rows')
    vocab=json.loads((data/'vocab.json').read_text()); keys={v['key']:i for i,v in enumerate(vocab)}
    if output.exists() and any(output.iterdir()): raise ValueError('Relation output must be empty')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=output.name+'.building-',dir=output.parent)); counts=Counter()
    with zipfile.ZipFile(archive) as z, sqlite3.connect(temp/'relations.sqlite') as db:
        db.execute('CREATE TABLE graphs (key TEXT PRIMARY KEY,payload BLOB)')
        db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY,split TEXT,graph_key TEXT)')
        for done,qid in enumerate(sorted(grouped,key=int),1):
            # Load query JSON, but access graph contents only for selected train/val rows.
            entries=json.loads(z.read(f'visualizer/casmi_{qid}.json'))
            lookup={(str(e['identifiers']['id']),int(e['research_metadata']['candidate_index'])):e for e in entries}
            exported=set()
            for row in grouped[qid]:
                identity=(row['spectrum_id'],row['candidate_index']); entry=lookup[identity]
                if row['source']=='experimental' and not entry['research_metadata']['is_target']:
                    raise ValueError('Experimental relation teacher must use the target DAG')
                key=json.dumps([row['split'],*identity],separators=(',',':'))
                if key not in exported:
                    graph=graph_record(entry,keys)
                    db.execute('INSERT INTO graphs VALUES (?,?)',(key,zlib.compress(json.dumps(graph,separators=(',',':')).encode())))
                    exported.add(key); counts['graphs/'+row['split']]+=1
                    counts['states/'+row['split']]+=len(graph['labels'])
                db.execute('INSERT INTO records VALUES (?,?,?)',(row['id'],row['split'],key))
                counts['records/'+row['split']]+=1
            db.commit()
            if done%10==0 or done==len(grouped): print(json.dumps(dict(relation_queries=done,total=len(grouped),counts=dict(counts))),flush=True)
    result=dict(schema_version=1,task='train-DAG typed relations',types=list(TYPES),peak_signature=signature(meta),
                partial=meta['partial'],test_graphs_exported=0,counts=dict(counts),
                hashes={'relations.sqlite':digest(temp/'relations.sqlite')},
                semantics='AIMe topology weak supervision; absent/corrupted relations are weak negatives, not chemical impossibility')
    (temp/'meta.json').write_text(json.dumps(result,indent=2)+'\n')
    if output.exists(): output.rmdir()
    temp.rename(output); return result


class RelationStore:
    def __init__(self,directory,peak_meta):
        self.directory=Path(directory); self.meta=json.loads((self.directory/'meta.json').read_text())
        if self.meta['schema_version']!=1 or self.meta['types']!=list(TYPES): raise ValueError('Relation schema mismatch')
        if self.meta['peak_signature']!=signature(peak_meta): raise ValueError('Relation/peak signatures differ')
        if self.meta.get('test_graphs_exported')!=0: raise ValueError('Test DAG supervision is forbidden')
        if digest(self.directory/'relations.sqlite')!=self.meta['hashes']['relations.sqlite']: raise ValueError('Relation checksum differs')
        self.db=sqlite3.connect((self.directory/'relations.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        if self.db.execute("SELECT COUNT(*) FROM records WHERE split NOT IN ('train','val')").fetchone()[0]:
            self.db.close(); self.db=None; raise ValueError('Forbidden test records in relation cache')
        self.cache=OrderedDict()

    def get(self,record):
        if record['split'] not in ('train','val'): raise ValueError('No test DAG access')
        row=self.db.execute('SELECT split,graph_key FROM records WHERE id=?',(record['id'],)).fetchone()
        if row is None or row[0]!=record['split']: raise ValueError('Missing or mismatched relation sample')
        key=row[1]
        if key not in self.cache:
            payload=self.db.execute('SELECT payload FROM graphs WHERE key=?',(key,)).fetchone()[0]
            self.cache[key]=GraphView(json.loads(zlib.decompress(payload)))
            if len(self.cache)>64: self.cache.popitem(last=False)
        self.cache.move_to_end(key); return self.cache[key]

    def close(self):
        if self.db is not None: self.db.close(); self.db=None

    def __del__(self):
        if getattr(self,'db',None) is not None: self.close()


def relation_examples(records,catalog,store,rng,pairs):
    samples=[]
    for record in records:
        view=store.get(record)
        condition=[record['collision_energy_nce']/100,record['precursor_mz']/1000,1 if record['precursor_type']=='[M+H]+' else -1]
        for a,b,flags in view.sample(rng,pairs):
            samples.append((a,b,condition,flags,1.))
            pool=[k for k in catalog.compatible(catalog.masses[b],record['precursor_type']) if k not in view.keys]
            if pool: samples.append((a,rng.choice(pool),condition,[0.]*4,.25))
    return samples


def make_training(records,catalog,store,rng,max_peaks,negatives,pairs,mode,context_peaks=8,relation_records=None):
    from casmi_peaks.data import training_targets
    support_seed=rng.getrandbits(64)
    targets=training_targets(records,catalog,rng,max_peaks,negatives)
    if targets is None: raise ValueError('Batch contains no covered targets')
    # Add observed support peaks without consulting their teacher labels. Their
    # candidate lists are mass-only samples and never receive structure loss.
    # This exposes the joint model to unannotated/uncertain context in training.
    original=targets['ids']; rows=[]
    for b,i in targets['positions'].tolist():
        rows.append((b,i))
    context=[]
    for b,record in enumerate(records):
        used={i for bb,i in rows if bb==b}
        remaining=[i for i in range(len(record['peaks'])) if i not in used and not record['precursor_peak'][i]]
        remaining.sort(key=lambda i:(-record['peaks'][i][1],record['peaks'][i][0]))
        added=0
        for i in remaining:
            if added>=context_peaks: break
            pool=catalog.compatible(record['peaks'][i][0],record['precursor_type'])
            if not pool: continue
            if len(pool)>negatives: pool=rng.sample(pool,negatives)
            context.append((b,i,pool))
            added+=1
    targets['supervised']=torch.ones(len(rows),dtype=torch.bool)
    if context:
        ids=sorted(set(original)|{k for _,_,pool in context for k in pool}); lookup={k:i for i,k in enumerate(ids)}
        width=max(targets['indices'].shape[1],max(len(pool) for _,_,pool in context)); n=len(rows)+len(context)
        updated=dict(ids=ids,positions=torch.tensor(rows+[(b,i) for b,i,_ in context]),
            indices=torch.zeros(n,width,dtype=torch.long),valid=torch.zeros(n,width,dtype=torch.bool),
            positive=torch.zeros(n,width,dtype=torch.bool),weights=torch.zeros(n,width),
            supervised=torch.tensor([True]*len(rows)+[False]*len(context)))
        old_width=targets['indices'].shape[1]
        updated['indices'][:len(rows),:old_width]=torch.tensor([lookup[k] for k in original])[targets['indices']]
        for key in ('valid','positive','weights'): updated[key][:len(rows),:old_width]=targets[key]
        for j,(_,_,pool) in enumerate(context,len(rows)):
            updated['indices'][j,:len(pool)]=torch.tensor([lookup[k] for k in pool]); updated['valid'][j,:len(pool)]=True
        targets=updated
    # Supervision lists must keep all positives, but message support must not
    # receive oracle positives. Independently sample support from the entire
    # mass-compatible pool using only observed m/z and mode, then union them
    # into the encoded graph batch. Only these IDs may send messages.
    support_rows=[]; candidate_rows=[]
    for row,(b,i) in enumerate(targets['positions'].tolist()):
        record=records[b]
        pool=catalog.compatible(record['peaks'][i][0],record['precursor_type'])
        support_rng=random.Random(support_seed+b*10007+i*1000003)
        support=support_rng.sample(pool,min(negatives,len(pool)))
        support_rows.append(set(support))
        existing=[targets['ids'][k] for k in targets['indices'][row,targets['valid'][row]].tolist()]
        candidate_rows.append(sorted(set(existing)|set(support)))
    ids=sorted({k for row in candidate_rows for k in row}); lookup={k:i for i,k in enumerate(ids)}
    n=len(candidate_rows); width=max(map(len,candidate_rows))
    rebuilt=dict(ids=ids,positions=targets['positions'],supervised=targets['supervised'],
        indices=torch.zeros(n,width,dtype=torch.long),valid=torch.zeros(n,width,dtype=torch.bool),
        positive=torch.zeros(n,width,dtype=torch.bool),weights=torch.zeros(n,width),support_valid=torch.zeros(n,width,dtype=torch.bool))
    for row,candidates in enumerate(candidate_rows):
        positive={targets['ids'][int(k)]:float(w) for k,w,p in zip(targets['indices'][row],targets['weights'][row],targets['positive'][row]) if p}
        for col,k in enumerate(candidates):
            rebuilt['indices'][row,col]=lookup[k]; rebuilt['valid'][row,col]=True
            rebuilt['support_valid'][row,col]=k in support_rows[row]
            if k in positive: rebuilt['positive'][row,col]=True; rebuilt['weights'][row,col]=positive[k]
    targets=rebuilt
    # Same-mass corruption is weak supervision, not chemical incompatibility.
    samples=relation_examples(records if relation_records is None else relation_records,catalog,store,rng,pairs)
    if not samples: samples=[(targets['ids'][0],targets['ids'][0],[0.,1.,1.],[0.]*4,0.)]
    old=targets['ids']; ids=sorted(set(old)|{x for a,b,_,_,_ in samples for x in (a,b)}); lookup={k:i for i,k in enumerate(ids)}
    remap=torch.tensor([lookup[k] for k in old]); targets['indices']=remap[targets['indices']]
    targets['ids']=ids; targets['mode']=mode
    targets['masses']=torch.tensor([[catalog.masses[ids[i]] for i in row] for row in targets['indices']],dtype=torch.float32)
    relation=dict(left=torch.tensor([lookup[a] for a,_,_,_,_ in samples]),right=torch.tensor([lookup[b] for _,b,_,_,_ in samples]),
        ma=torch.tensor([catalog.masses[a] for a,_,_,_,_ in samples],dtype=torch.float32),
        mb=torch.tensor([catalog.masses[b] for _,b,_,_,_ in samples],dtype=torch.float32),
        conditions=torch.tensor([c for _,_,c,_,_ in samples],dtype=torch.float32),
        labels=torch.tensor([y for _,_,_,y,_ in samples],dtype=torch.float32),
        weights=torch.tensor([w for _,_,_,_,w in samples],dtype=torch.float32))
    relation['head_weights']=torch.where((relation['labels']>0)|(relation['weights'][:,None]<1),1.,.25)
    return targets,relation
