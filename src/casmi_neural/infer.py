from dataclasses import asdict
import gzip
import json
from pathlib import Path

import torch

from .data import PrepareConfig,SpectrumDAGDataset,collate,entry_record,to_device,visibility
from .flow import decompose
from .model import ModelConfig,SpectrumDAGModel
from .mass_check import node_mass_bounds,check_peak
from .train import dataset_signature
from .runtime import configure_cuda


def infer(checkpoint,output,data=None,sample_id=None,entry_json=None,entry_index=0,source='experimental',device_name='cpu',spectrum_json=None):
    checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=True)
    prep=PrepareConfig(**checkpoint['dataset_signature']['config'])
    if entry_json:
        path=Path(entry_json)
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
            content=json.load(f)
        entry=content[entry_index] if isinstance(content,list) else content.get('entry',content)
        if spectrum_json:
            peaks=json.loads(Path(spectrum_json).read_text())
            entry=dict(entry);entry[f'{source}_ms2']=peaks.get('peaks') if isinstance(peaks,dict) else peaks
        record=entry_record(entry,source,prep)
        record['id']=0
        record['info']={'source':source,'entry_index':entry_index,'spectrum_id':entry.get('identifiers',{}).get('id','custom')}
    else:
        if data is None or sample_id is None:raise ValueError('Supply --data/--sample-id or --entry-json')
        dataset=SpectrumDAGDataset(data,['predicted','experimental'],'all')
        if dataset_signature(dataset.meta)!=checkpoint['dataset_signature']:raise ValueError('Data signature mismatch')
        found=[i for i,r in enumerate(dataset.index) if r['id']==sample_id]
        if not found:raise KeyError(f'Sample id {sample_id} absent from index.jsonl')
        record=dataset[found[0]]
    device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name)
    torch.set_num_threads(2)
    if device.type=='cuda':configure_cuda(checkpoint['training_config'])
    model=SpectrumDAGModel(ModelConfig(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    batch=to_device(visibility(collate([record]),0.,0),device)
    with torch.no_grad():
        result=model(batch);flow=result['posterior'];parts=decompose(batch,flow)
    nb=int(batch['others'][0]);source_nodes,child_nodes=record['edge_index']
    flow_cpu={k:v.cpu() for k,v in flow.items()}
    predicted=flow_cpu['spectrum'];emission=flow_cpu['emission']
    edge_parts=parts['edge_by_bin'].cpu();node_parts=parts['node_by_bin'].cpu()
    # This invariant checks graph-flow decomposition, not chemical correctness.
    conservation=float((edge_parts.sum(1)-flow_cpu['edge_flow']).abs().max()) if len(edge_parts) else 0.
    observed_bins=sorted({int(b) for b in record['peak_bin'] if int(b)<nb})
    unobserved_bins=sorted(set(range(nb))-set(observed_bins))
    residual_edge=edge_parts[:,unobserved_bins].sum(1)
    residual_node=node_parts[:,unobserved_bins].sum(1)
    residual_signal=float(predicted[unobserved_bins].sum())
    observed_edge=edge_parts[:,observed_bins].sum(1)
    partition_error=float((observed_edge+residual_edge-flow_cpu['edge_flow']).abs().max()) if len(edge_parts) else 0.
    nids=record['nids'];annotations=[]
    node_masses,mass_bounds,mass_source=node_mass_bounds(record)
    for i,(mz,intensity) in enumerate(record['peaks'].tolist()):
        b=int(record['peak_bin'][i]);group=torch.nonzero(record['peak_bin']==b).flatten().tolist()
        mass_checks=check_peak(node_masses,mass_bounds,mz,prep)
        compatible=[v for v,r in enumerate(mass_checks) if r['mass_compatibility']=='compatible']
        boundary=[v for v,r in enumerate(mass_checks) if r['mass_compatibility']=='boundary_uncertain']
        selected=[v for v in range(len(nids)) if int(record['node_bin'][v])==b] if b!=nb else []
        incompatible=[v for v in selected if mass_checks[v]['mass_compatibility']=='incompatible']
        alternatives=[v for v in compatible if v not in selected]
        item={'peak_index':i,'mz':mz,'intensity':intensity,'mass_bin':b,
              'shares_mass_bin_with_peak_indices':group,'unexplained':b==nb,
              'assignment_scope':'model_mass_bin',
              'mass_compatible_nids':[nids[v] for v in compatible],
              'mass_boundary_uncertain_nids':[nids[v] for v in boundary],
              'alternative_compatible_nids_outside_bin':[nids[v] for v in alternatives],
              'mass_incompatible_bin_nids':[nids[v] for v in incompatible],
              'mass_bin_missed_compatible_fragment':b==nb and bool(compatible),
              'terminal_fragments':[],'nodes':[],'edges':[],
              'assignment_status':'no_compatible_fragment' if b==nb else 'negligible_model_support' if float(predicted[b])<=1e-12 else 'assigned'}
        if item['mass_bin_missed_compatible_fragment']:
            item['assignment_status']='mass_bin_missed_compatible_fragment'
        elif incompatible:
            item['assignment_status']='mass_bin_contains_incompatible_fragments'
        elif any(v in boundary for v in selected):
            item['assignment_status']='mass_boundary_requires_review'
        if b!=nb and float(predicted[b])>1e-12:
            denominator=float(predicted[b])
            for v in range(len(nids)):
                if int(record['node_bin'][v])==b:
                    item['terminal_fragments'].append({'nid':nids[v],'smiles':record['smiles'][v],
                         'formula':record['formulas'][v],'conditional_contribution':float(emission[v])/denominator,
                         **mass_checks[v]})
                probability=float(node_parts[v,b])/denominator
                if probability>0:
                    item['nodes'].append({'nid':nids[v],'conditional_path_occupancy':probability})
            for e in range(len(source_nodes)):
                value=float(edge_parts[e,b])
                if value>0:
                    item['edges'].append({'source_nid':nids[int(source_nodes[e])],
                       'target_nid':nids[int(child_nodes[e])], 'signal_flow':value,
                       'conditional_path_occupancy':value/denominator})
            item['predicted_bin_signal']=denominator
            item['observed_bin_signal']=float(record['target'][b])
        annotations.append(item)
    report={'sample':record['info'],'checkpoint_epoch':checkpoint['epoch'],
        'interpretation':'Model-conditioned signal attribution, not calibrated chemical confidence or experimentally proven paths.',
        'node_mass_source':mass_source,
        'mass_check_note':'Flow is decomposed by model mass bin. Check each terminal mass separately; warnings do not discard or renormalize flow. Older caches use conservative rounding bounds.',
        'mass_incompatible_bin_peak_count':sum(bool(r['mass_incompatible_bin_nids']) for r in annotations),
        'mass_bin_missed_compatible_peak_count':sum(r['mass_bin_missed_compatible_fragment'] for r in annotations),
        'mass_bin_note':'Peaks in one model mass bin share one readout. This is a modelling approximation, not proof of physical indistinguishability. Do not sum duplicate explanations; OTHER pools unrelated unmatched peaks.',
        'training_intensity_power':prep.intensity_power,'mass_tolerance':asdict(prep),
        'spectrum_total':float(predicted.sum()),'other_prediction':float(predicted[-1]),
        'other_target':float(record['target'][-1]),'edge_decomposition_max_abs_error':conservation,
        'observed_plus_unobserved_partition_error':partition_error,
        'unobserved_dag_signal':residual_signal,
        'unobserved_mass_bins':[{'mass_bin':b,'mz':float(record['bin_masses'][b]),'predicted_signal':float(predicted[b])}
                                for b in unobserved_bins],
        'unobserved_subnetwork':{
            'nodes':[{'nid':nids[v],'signal_occupancy':float(residual_node[v])} for v in range(len(nids)) if float(residual_node[v])>0],
            'edges':[{'source_nid':nids[int(i)],'target_nid':nids[int(j)],'signal_flow':float(residual_edge[e])}
                     for e,(i,j) in enumerate(zip(source_nodes,child_nodes)) if float(residual_edge[e])>0]},
        'dag_nodes':[{'nid':nid,'smiles':s,'formula':f,'emission':float(emission[v]),
                     'stop_probability':float(flow_cpu['stop'][v]),'reach':float(flow_cpu['reach'][v])}
                    for v,(nid,s,f) in enumerate(zip(nids,record['smiles'],record['formulas']))],
        'dag_edges':[{'source_nid':nids[int(i)],'target_nid':nids[int(j)],
                     'transition_probability':float(flow_cpu['transition'][e]),'signal_flow':float(flow_cpu['edge_flow'][e])}
                    for e,(i,j) in enumerate(zip(source_nodes,child_nodes))],
        'peaks':annotations}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'peaks':len(annotations),'unexplained_peaks':sum(r['unexplained'] for r in annotations),
                      'mass_incompatible_bin_peaks':report['mass_incompatible_bin_peak_count'],
                      'mass_bin_missed_compatible_peaks':report['mass_bin_missed_compatible_peak_count'],
                      'edge_decomposition_error':conservation}),flush=True)
