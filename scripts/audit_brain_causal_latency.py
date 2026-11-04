"""Optimistic input-to-descending spike latency on the complete directed graph.

This is a causal lower bound from implemented event order, not a biological
reaction-time measurement or a proof that a reachable neuron will fire.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.connectome import load_connectome, file_sha256
from fly_touhou.whole_brain_learning import graph_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    output=options.output.resolve()
    if output.exists():
        raise FileExistsError('Causal audit evidence is never overwritten')
    checkpoint=torch.load(options.checkpoint,map_location='cpu',weights_only=True)
    graph=load_connectome(options.data,max_neurons=None)
    if graph.n_neurons!=138639 or graph.n_edges!=15091983 or graph_fingerprint(graph)!=checkpoint['graph_fingerprint']:
        raise ValueError('Causal audit must use the exact checkpoint full directed graph')
    config=checkpoint['model_config']
    inputs=np.asarray(config['input_indices'],dtype=np.int32)
    outputs=np.asarray(config['output_indices'],dtype=np.int32)
    adjacency=csr_matrix((graph.signed_weights.copy(),(graph.pre,graph.post)),
        shape=(graph.n_neurons,graph.n_neurons))
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.)
    distances=dijkstra(adjacency,directed=True,indices=inputs,unweighted=True,min_only=True)[outputs]
    dt=float(config['dt_ms'])
    delay_steps=round(config['parameters']['delay_ms']/dt)
    # Changed input event at step0 can change its input neuron's spike at
    # step1. A changed presynaptic spike enters the delay queue, then the
    # postsynaptic conductance AFTER thresholding; the next integration can
    # first change a postsynaptic spike. Thus each edge costs delay_steps+1.
    earliest_step=1+(delay_steps+1)*distances
    head=checkpoint['trainable_parameters']['action_readout.weight'].numpy()
    head_active=np.any(head!=0,axis=0)
    summaries={}
    for substeps in (25,50,100,200):
        reachable=earliest_step<substeps
        summaries[str(substeps)]={'neural_ms_per_decision':substeps*dt,
            'descending_neurons':len(outputs),'optimistically_causally_reachable_in_first_decision':int(reachable.sum()),
            'head_columns_optimistically_reachable_in_first_decision':int((reachable&head_active).sum()),
            'reachable_fraction':float(reachable.mean())}
    finite=distances[np.isfinite(distances)]
    result={'schema':'whole-brain-causal-latency-audit-v1',
        'checkpoint':str(options.checkpoint.resolve()),'checkpoint_sha256':file_sha256(options.checkpoint),
        'script_sha256':file_sha256(__file__),'brain_code_sha256':file_sha256(ROOT/'src/fly_touhou/whole_brain.py'),
        'neurons':graph.n_neurons,'edge_records':graph.n_edges,'directed':True,
        'input_ports':len(inputs),'output_ports':len(outputs),'dt_ms':dt,'delay_steps':delay_steps,
        'descendants_unreachable':int(np.isinf(distances).sum()),
        'edge_distance_quantiles':dict(zip(('minimum','p10','median','p90','maximum'),
            np.quantile(finite,[0,.1,.5,.9,1]).tolist())),
        'per_output':[{'index':int(index),'node_id':str(graph.node_ids[index]),
            'minimum_directed_edges':None if not np.isfinite(distance) else int(distance),
            'optimistic_first_changed_spike_step':None if not np.isfinite(step) else int(step),
            'head_column_nonzero':bool(active)}
            for index,distance,step,active in zip(outputs,distances,earliest_step,head_active)],
        'clock_comparison':summaries,
        'scope':['Only direct causal reachability from a newly changed current observation.',
            'Signed inhibition, threshold, refractory state and stochastic input waits can add latency or prevent any spike.',
            'The 20ms descending rate filter can further attenuate a fast action change.',
            'Earlier observations can support anticipation; this does not prove a game action is impossible.',
            'No claim about living fly reaction times or native-game success.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('edge_distance_quantiles','clock_comparison')},ensure_ascii=False))


if __name__=='__main__':
    main()
