"""Deployment only needs V2 weights, the train catalog and observed spectrum."""
import json
from pathlib import Path
from casmi_peaks.data import Catalog,digest,tolerance
from casmi_peaks.infer import read_spectrum
from casmi_peaks.engine import device_context
from .engine import load_model
from .evaluate import load_calibration
from .predict import rank_record


def infer(checkpoint,data,input_path,output,topk=10,device_name='auto',calibration=None,strength=None):
    device,rank,world=device_context(device_name)
    if world!=1: raise ValueError('Use a single process for deployment')
    record=read_spectrum(json.loads(Path(input_path).read_text()))
    catalog=Catalog(data); model,ckpt=load_model(checkpoint,data,device); settings=ckpt['config']['joint']
    if calibration: settings=load_calibration(calibration,checkpoint,settings,ckpt['dataset_signature'])
    if strength is not None: settings={**settings,'strength':float(strength)}
    if not 1<=topk<=settings['candidate_k']: raise ValueError('topk must be within the configured shortlist')
    prediction=rank_record(model,record,catalog,device,settings); peaks=[]
    for i,(mz,intensity) in enumerate(record['peaks']):
        candidates=[]
        for k in prediction['joint_rank'][i][:topk]:
            label=catalog.labels[k]
            candidates.append(dict(catalog_id=k,exported_structure_smiles=label['smiles'],reported_ion_formula=label['formula'],
                precursor_type=label['ion'],catalog_mz=label['mz'],mass_error_da=abs(label['mz']-mz),
                unary_logit=prediction['unary_scores'][i][k],joint_logit=prediction['joint_scores'][i][k],
                formula_smiles_consistent=label['formula_smiles_consistent']))
        precursor=abs(mz-record['precursor_mz'])<=tolerance(mz,catalog.cfg)
        peaks.append(dict(peak_index=i,mz=mz,intensity=intensity,mass_compatible_candidates=len(prediction['pools'][i]),
            unary_top1_catalog_id=prediction['unary_rank'][i][0] if prediction['unary_rank'][i] else None,
            candidates=candidates,status='precursor_peak_not_annotated' if precursor else
                'ranked_unconfirmed_candidates' if candidates else 'no_mass_compatible_training_catalog_candidate'))
    packet=dict(task='peak_fragment_joint_v2',checkpoint_epoch=ckpt['epoch'],checkpoint_sha256=digest(checkpoint),
        partial_catalog=ckpt['dataset_signature']['partial'],input_conditions={k:v for k,v in record.items() if k!='peaks'},
        limited_training=ckpt.get('limited_training',False),runtime_implementation_revision=2,
        weights_implementation_revision=ckpt.get('implementation_revision',1),
        joint_settings=settings,peaks=peaks,
        interpretation='Joint ranking from learned AIMe relations; no query DAG. Scores are uncalibrated, labels are not confirmed ion structures, and mass-compatible candidates do not establish OOV explanations.')
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),peaks=len(peaks),with_candidates=sum(bool(p['candidates']) for p in peaks))),flush=True)
    return packet

