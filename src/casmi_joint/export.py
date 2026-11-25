"""Portable test-set viewer export; predictions and teacher display stay separate."""
from collections import Counter,defaultdict
import base64
import gzip
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import re
import zipfile
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from casmi_peaks.chem import structure
from casmi_peaks.data import Catalog,PeakDataset,digest,signature,spectrum_batch
from casmi_peaks.engine import device_context,move
from casmi_peaks.infer import read_spectrum
from .predict import observed_pools,encode_observed_catalog,prepare_record,rank_prepared

KS=(1,3,5,10)


def peak_status(peak,ranker='joint',topk=10):
    if peak['is_precursor']: return 'precursor'
    truth=set(peak['teacher_keys'])
    if not truth: return 'unannotated'
    if not peak['teacher_covered']: return 'gate_miss' if peak.get('teacher_in_vocab',False) else 'oov'
    if ranker!='frequency' and not peak.get('predictions_available',True): return 'pending'
    ranking=peak['rankings'].get(ranker,[])
    best=peak.get('teacher_best_rank',{}).get(ranker)
    if best is None: best=next((i+1 for i,x in enumerate(ranking) if x['key'] in truth),None)
    if best==1: return 'hit1'
    if best is not None and best<=topk: return 'hitk'
    return 'miss'


def summarize_bundle(bundle):
    count=Counter(); hits={k:Counter() for k in ('unary','joint','frequency')}; queries=set()
    for spectrum in bundle['spectra']:
        count['spectra']+=1
        for peak in spectrum['peaks']:
            count['total_peaks']+=1
            if peak['is_precursor']: count['precursor_peaks']+=1; continue
            count['eligible_fragment_peaks']+=1
            truth=set(peak['teacher_keys'])
            if not truth: count['unannotated_fragment_peaks']+=1; continue
            queries.add(str(spectrum['qid']))
            count['teacher_annotated_peaks']+=1
            count['vocab_covered_peaks']+=bool(peak['teacher_covered'])
            count['uncovered_teacher_peaks']+=not peak['teacher_covered']
            count['ambiguous_teacher_peaks']+=len(truth)>1
            count['shortlist_covered_peaks']+=bool(peak.get('teacher_best_rank',{}).get('joint'))
            for ranker in hits:
                for k in KS:
                    # best_rank is computed from full unary/frequency retrieval
                    # or the complete joint shortlist, independent of display K.
                    best=peak.get('teacher_best_rank',{}).get(ranker)
                    if best is None:
                        row=peak['rankings'].get(ranker,[])
                        best=next((j+1 for j,v in enumerate(row) if v['key'] in truth),None)
                    hits[ranker][k]+=best is not None and best<=k
    coverage={key:int(count[key]) for key in ('spectra','total_peaks','precursor_peaks','eligible_fragment_peaks',
        'unannotated_fragment_peaks','teacher_annotated_peaks','vocab_covered_peaks','uncovered_teacher_peaks',
        'ambiguous_teacher_peaks','shortlist_covered_peaks')}
    coverage['queries_with_annotations']=len(queries)
    coverage['teacher_peak_coverage']=count['teacher_annotated_peaks']/max(1,count['eligible_fragment_peaks'])
    coverage['vocabulary_coverage_of_annotated']=count['vocab_covered_peaks']/max(1,count['teacher_annotated_peaks'])
    result={'coverage':coverage}
    for ranker in hits:
        available=ranker=='frequency' or bundle.get('mode')=='predictions'
        result[ranker]={}
        for k in KS:
            result[ranker][f'hits{k}']=int(hits[ranker][k]) if available else None
            for suffix,denominator in [('all_annotated',count['teacher_annotated_peaks']),('vocab_covered',count['vocab_covered_peaks'])]:
                result[ranker][f'teacher_top{k}_{suffix}']=hits[ranker][k]/denominator if available and denominator else None
    return result


class TeacherResolver:
    """Recover labels from each record's own candidate/energy DAG.

    Node IDs are never treated as list positions. The resolved structures are
    display/evaluation metadata and are not passed to student prediction.
    """
    def __init__(self,archive): self.archive=Path(archive); self.precursors={}

    def resolve(self,records):
        result={}; grouped=defaultdict(list)
        for record in records: grouped[str(record['qid'])].append(record)
        with zipfile.ZipFile(self.archive) as z:
            for qid,rows in grouped.items():
                entries=json.loads(z.read(f'visualizer/casmi_{qid}.json'))
                lookup={(str(e['identifiers']['id']),int(e['research_metadata']['candidate_index'])):e for e in entries}
                for record in rows:
                    entry=lookup[(record['spectrum_id'],record['candidate_index'])]
                    predicted=record['source']=='predicted'
                    if not predicted and not entry['research_metadata']['is_target']: raise ValueError('Display teacher must be the correct precursor DAG')
                    parent=entry.get('pred_mol_metadata' if predicted else 'true_mol_metadata',{})
                    if parent.get('smiles'):
                        self.precursors[record['id']]=dict(smiles=parent['smiles'],formula=parent.get('molecular_formula',''),
                            svg=structure_svg(parent['smiles']),reference='generating_precursor' if predicted else 'known_correct_precursor')
                    wanted={key for target in record['targets'] for key,_ in target}
                    recovered=set()
                    for node in entry['dag_nodes']:
                        if not node['in_nids']: continue
                        try: label=structure(node['SMILES'],node['formula'],record['precursor_type'])
                        except ValueError: continue
                        if label['key'] in wanted:
                            result[label['key']]={**label,'mz':float(node['mass'])}
                            recovered.add(label['key'])
                    if wanted-recovered: raise ValueError(f'Unable to recover teacher structure(s) for {record["id"]}; check RDKit/cache/archive identity')
        return result


@lru_cache(maxsize=20000)
def structure_svg(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError('Cannot draw invalid structure')
    draw=rdMolDraw2D.MolDraw2DSVG(300,180); options=draw.drawOptions()
    options.clearBackground=False
    options.updateAtomPalette({i:(.73,.88,.93) for i in range(1,119)})
    options.updateAtomPalette({7:(.48,.72,1.),8:(1.,.57,.62),16:(.96,.82,.43),9:(.39,.88,.74),17:(.39,.88,.74),35:(.78,.62,.91)})
    options.bondLineWidth=1.6; options.padding=.12
    rdMolDraw2D.PrepareAndDrawMolecule(draw,mol); draw.FinishDrawing()
    svg=draw.GetDrawingText(); return svg[svg.index('<svg'):]


def render_html(bundle,template_dir=None):
    directory=Path(template_dir) if template_dir else Path(__file__).resolve().parents[2]/'test_atlas'
    html=(directory/'index.html').read_text(); css=(directory/'style.css').read_text(); js=(directory/'app.js').read_text()
    payload=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    encoded=base64.b64encode(gzip.compress(payload,mtime=0)).decode()
    replacement={'STYLE':'<style>'+css.replace('</style','<\\/style')+'</style>',
        'SCRIPT':'<script>'+js.replace('</script','<\\/script')+'</script>',
        'DATA':'<script id="atlas-data" type="application/octet-stream">'+encoded+'</script>'}
    for marker,content in replacement.items():
        pattern=f'<!-- ATLAS_{marker} -->[\\s\\S]*?<!-- /ATLAS_{marker} -->'
        html,n=re.subn(pattern,lambda _:content,html)
        if n!=1: raise ValueError(f'Missing/duplicate HTML {marker} placeholder')
    return html


def _compare_reference(bundle,reference_path):
    reference=json.loads(Path(reference_path).read_text()); provenance=bundle['provenance']
    if reference.get('sources',['experimental'])!=[provenance.get('source','experimental')]:
        raise ValueError('Reference metrics spectrum source differs')
    if reference['checkpoint_sha256']!=provenance['checkpoint_sha256']: raise ValueError('Reference metrics checkpoint differs')
    if reference['split']!=provenance['split'] or reference['dataset_signature']!=provenance['dataset_signature']:
        raise ValueError('Reference metrics data/split differs')
    rm=reference['metrics']; settings=rm['settings']
    if settings!=provenance['joint_settings']: raise ValueError('Reference joint settings differ; use the same calibration/strength')
    got=bundle['metrics']; expected_joint=rm['joint_by_strength'][f'{settings["strength"]:g}']; mismatches=[]
    fields=('spectra','total_peaks','precursor_peaks','eligible_fragment_peaks','teacher_annotated_peaks',
            'vocab_covered_peaks','unannotated_fragment_peaks','shortlist_covered_peaks')
    for key in fields:
        if got['coverage'][key]!=rm['coverage'][key]: mismatches.append(key)
    for k in KS:
        key=f'teacher_top{k}_all_annotated'
        for ranker,expected in [('unary',rm['unary'][key]),('joint',expected_joint[key]),
                                ('frequency',rm['coverage'][f'training_frequency_top{k}_all_annotated'])]:
            if abs(got[ranker][key]-expected)>1e-12: mismatches.append(f'{ranker}.{key}')
    if mismatches: raise ValueError('Per-peak export differs from reported metrics: '+', '.join(mismatches)+'. No final HTML was written; verify device/config/checkpoint.')
    return dict(status='matched',metrics_sha256=digest(reference_path),checked_counts=list(fields),checked_topk=list(KS))


@torch.no_grad()
def export_test(data,archive,output,checkpoint=None,calibration=None,reference_metrics=None,topk=10,
                split='test',device_name='auto',teacher_only=False,limit=None,template_dir=None,json_output=None,source='experimental'):
    from .engine import load_model
    from .evaluate import load_calibration
    if topk<10: raise ValueError('Export at least Top-10 for consistent visual metrics')
    device,_,world=device_context('cpu' if teacher_only else device_name)
    if world!=1: raise ValueError('Portable viewer export uses one process')
    if source not in ('predicted','experimental'): raise ValueError('Invalid spectrum source')
    dataset=PeakDataset(data,[source],split,limit); catalog=Catalog(data)
    records=[dataset[i] for i in range(len(dataset))]; dataset.close()
    if not records: raise ValueError('Empty export split')
    if digest(archive)!=dataset.meta['reference_signature']['archive_sha256']: raise ValueError('Archive identity mismatch')
    # Prediction receives only whitelisted observed spectrum and conditions.
    students=[read_spectrum(record) for record in records]
    if any(student['peaks']!=record['peaks'] for student,record in zip(students,records)):
        raise ValueError('Cache peaks must be sorted, finite and positive; refusing misaligned peak predictions')
    model=ckpt=settings=embeddings=None
    if not teacher_only:
        if checkpoint is None: raise ValueError('A real V2 checkpoint is required; teacher-only preview is explicit')
        model,ckpt=load_model(checkpoint,data,device); settings=ckpt['config']['joint']
        if calibration: settings=load_calibration(calibration,checkpoint,settings,ckpt['dataset_signature'])
        if topk>settings['candidate_k']: raise ValueError('Display Top-k exceeds model shortlist')
        embeddings=encode_observed_catalog(model,catalog,students,device,256)
    resolver=TeacherResolver(archive); teacher_labels=resolver.resolve(records); structures={}; spectra=[]
    def register(key):
        if key in structures: return
        in_vocab=key in catalog.keys
        label=catalog.labels[catalog.keys[key]] if in_vocab else teacher_labels.get(key)
        if label is None: raise ValueError('Missing display label')
        structures[key]=dict(smiles=label['smiles'],formula=label['formula'],ion=label['ion'],mz=float(label['mz']),
            svg=structure_svg(label['smiles']),in_vocab=in_vocab,catalog_id=int(catalog.keys[key]) if in_vocab else None,
            train_frequency=int(label.get('train_peak_occurrences',0)) if in_vocab else None,
            formula_smiles_consistent=bool(label['formula_smiles_consistent']))
    for start in range(0,len(records),8):
        queries=model.encode_spectrum(move(spectrum_batch(students[start:start+8]),device)).cpu().numpy() if model else None
        for b,record in enumerate(records[start:start+8]):
            student=students[start+b]; prepared=prediction=None
            if model:
                prepared=prepare_record(model,student,catalog,device,settings,embeddings,queries[b])
                prediction=rank_prepared(model,prepared,settings)
            pools=prediction['pools'] if prediction else observed_pools(student,catalog)
            peaks=[]
            for i,(mz,intensity) in enumerate(record['peaks']):
                truth=[key for key,_ in record['targets'][i]]
                known={catalog.keys[key] for key in truth if key in catalog.keys}
                pool=pools[i]
                frequency=sorted(pool,key=lambda k:(-catalog.freq[k],abs(catalog.masses[k]-mz),k))
                ranking_ids={'frequency':frequency,'unary':prediction['unary_rank'][i] if prediction else [],
                             'joint':prediction['joint_rank'][i] if prediction else []}
                rankings={}; best={}; teacher_ranks={key:{} for key in truth}
                for ranker,ids in ranking_ids.items():
                    positions={k:j+1 for j,k in enumerate(ids)}
                    best[ranker]=min((positions[k] for k in known if k in positions),default=None)
                    for key in truth: teacher_ranks[key][ranker]=positions.get(catalog.keys.get(key,-1))
                    rows=[]
                    for k in ids[:topk]:
                        key=catalog.labels[k]['key']; register(key)
                        score=float(catalog.freq[k]) if ranker=='frequency' else prediction[ranker+'_scores'][i][k]
                        rows.append(dict(key=key,score=score))
                    rankings[ranker]=rows
                for key in truth: register(key)
                peak=dict(mz=mz,intensity=intensity,is_precursor=bool(record['precursor_peak'][i]),teacher_keys=truth,
                    teacher_in_vocab=bool(known),teacher_covered=bool(known.intersection(pool)),
                    teacher_best_rank=best,teacher_ranks=teacher_ranks,candidate_count=len(pool),rankings=rankings,
                    predictions_available=not teacher_only)
                peak['status']=peak_status(peak); peaks.append(peak)
            spectra.append(dict(id=record['id'],qid=str(record['qid']),nce=record['collision_energy_nce'],source=source,
                spectrum_id=record['spectrum_id'],candidate_index=record['candidate_index'],
                precursor_mz=record['precursor_mz'],ion=record['precursor_type'],precursor=resolver.precursors.get(record['id']),peaks=peaks))
        print(json.dumps(dict(exported_spectra=min(start+8,len(records)),total=len(records),display_structures=len(structures))),flush=True)
    bundle=dict(schema_version=1,mode='teacher_only' if teacher_only else 'predictions',
        provenance=dict(split=split,source=source,protocol=dataset.meta['reference_signature']['config']['split'],
            checkpoint_sha256=digest(checkpoint) if checkpoint else None,checkpoint_epoch=ckpt['epoch'] if ckpt else None,
            dataset_signature=signature(dataset.meta),joint_strength=settings['strength'] if settings else None,
            joint_settings=settings,partial_data=dataset.meta['partial'] or limit is not None or bool(ckpt and ckpt.get('limited_training')),
            display_topk=topk,teacher_semantics=dataset.meta['label_semantics'],
            note='AIMe teacher candidates are hypotheses, not experimentally confirmed fragment structures; model scores are not calibrated chemical confidence.'),
        structures=structures,spectra=spectra)
    bundle['metrics']=summarize_bundle(bundle)
    if reference_metrics:
        if teacher_only: raise ValueError('Teacher-only preview cannot verify neural prediction metrics')
        bundle['provenance']['reference_check']=_compare_reference(bundle,reference_metrics)
    else: bundle['provenance']['reference_check']=dict(status='not_supplied')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    html=render_html(bundle,template_dir); temporary=output.with_suffix(output.suffix+'.tmp'); temporary.write_text(html); temporary.replace(output)
    if json_output:
        path=Path(json_output); path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        path.write_bytes(gzip.compress(content,mtime=0) if str(path).endswith('.gz') else content)
    print(json.dumps(dict(output=str(output),spectra=len(spectra),queries=len({s['qid'] for s in spectra}),
        bytes=output.stat().st_size,mode=bundle['mode'],reference_check=bundle['provenance']['reference_check']['status'])),flush=True)
    return bundle
