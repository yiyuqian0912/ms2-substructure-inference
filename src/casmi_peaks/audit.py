"""Measure teacher and train-catalog coverage without fitting or consulting a student."""
from collections import Counter
import json
import math
from pathlib import Path
from .data import Catalog,PeakDataset


def audit(data,output):
    catalog=Catalog(data,load_features=False);results={}
    for split in ['train','val','test']:
        dataset=PeakDataset(data,['experimental'],split);c=Counter();counts=[];qids=set()
        for record in dataset:
            c['spectra']+=1;qids.add(record['qid']);signal=annotated_signal=0.
            for i,(mz,intensity) in enumerate(record['peaks']):
                c['total_peaks']+=1
                if record['precursor_peak'][i]:c['precursor_peaks']+=1;continue
                c['eligible_fragment_peaks']+=1;signal+=math.sqrt(intensity)
                labels=record['targets'][i]
                if not labels:continue
                c['teacher_annotated_peaks']+=1;annotated_signal+=math.sqrt(intensity)
                ids=catalog.compatible(mz,record['precursor_type']);counts.append(len(ids))
                positives={catalog.keys[k] for k,_ in labels if k in catalog.keys}
                c['catalog_covered_peaks']+=bool(positives.intersection(ids))
                c['ambiguous_teacher_peaks']+=len(labels)>1
            c['signal_coverage_sum']+=annotated_signal/signal if signal else 0.
        dataset.close()
        result={k:int(c[k]) for k in ['spectra','total_peaks','precursor_peaks','eligible_fragment_peaks','teacher_annotated_peaks','catalog_covered_peaks','ambiguous_teacher_peaks']}
        result.update(queries=len(qids),teacher_peak_coverage=c['teacher_annotated_peaks']/max(1,c['eligible_fragment_peaks']),
            catalog_coverage_of_annotated=c['catalog_covered_peaks']/max(1,c['teacher_annotated_peaks']),
            teacher_signal_coverage_spectrum_mean=c['signal_coverage_sum']/max(1,c['spectra']),
            mean_mass_compatible_candidates=sum(counts)/max(1,len(counts)),max_mass_compatible_candidates=max(counts,default=0))
        results[split]=result
    report=dict(vocabulary=len(catalog.labels),train_only_catalog=True,
        catalog_mass_spread_max=float((catalog.maximum-catalog.minimum).max()),
        formula_smiles_disagreement_labels=sum(not v['formula_smiles_consistent'] for v in catalog.labels),
        experimental=results,interpretation='Coverage of AIMe weak labels, not chemical accuracy; root/precursor windows excluded from fragment denominators.')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True);return report
