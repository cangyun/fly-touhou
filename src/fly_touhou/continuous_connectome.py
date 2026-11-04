"""Continuous scalar-per-neuron RNN on the complete pinned signed connectome.

This is a NEW family, not a LIF numerical backend or an old-checkpoint resume.
No spikes, neural RNG, refractory state, delay queue or100 microsteps exist.
One explicit activity state is carried across decisions; only callers knowing
a genuine new native game may reset it. No job/model/data is started on import.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .connectome import ConnectomeGraph, EXPECTED_NEURONS, load_connectome, file_sha256
from .continuous_sparse import TrainableDirectedSparse

SCHEMA = 'continuous-complete-signed-connectome-RNN-v1'
EXPECTED_EDGES = 15_091_983
PINNED_GRAPH_FINGERPRINT = '7b6a3471f9ab7ec1ec6ef3bd9e53c110a18d2b94d0d541d95cb5e2e6e8705b22'
ANNOTATIONS_SHA256 = '30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6'
LEGACY_DN_MANIFEST_SHA256 = '2af70522365ea9168466c438d99429be62be4df108dca073b34f6c637c7e7baf'


def graph_fingerprint(graph):
    digest = sha256(json.dumps(list(graph.node_ids),separators=(',',':')).encode())
    for value,dtype in ((graph.source_indices,'<i4'),(graph.pre,'<i4'),
            (graph.post,'<i4'),(graph.signed_weights,'<f4')):
        digest.update(memoryview(np.ascontiguousarray(value,dtype=dtype)).cast('B'))
    return digest.hexdigest()


@dataclass(frozen=True)
class ContinuousConnectomeState:
    activity: torch.Tensor

    def __post_init__(self):
        if (not torch.is_tensor(self.activity) or self.activity.dtype != torch.float32
                or self.activity.ndim != 2 or min(self.activity.shape) < 1):
            raise ValueError('Continuous FP32 activity[B,N] is the complete state')

    def detach(self): return type(self)(self.activity.detach())
    def to(self,device,*,non_blocking=False):
        return type(self)(self.activity.to(device=device,non_blocking=non_blocking))
    @staticmethod
    def pack(states): return pack_continuous_states(states)
    def unpack(self,count=None): return unpack_continuous_state(self,count=count)


def pack_continuous_states(states):
    states = tuple(states)
    if not states or any(not isinstance(s,ContinuousConnectomeState) for s in states):
        raise ValueError('Nonempty continuous-state list required')
    first = states[0].activity
    if any(s.activity.shape[1:] != first.shape[1:] or s.activity.device != first.device for s in states):
        raise ValueError('All lanes need the same graph and device')
    return ContinuousConnectomeState(torch.cat([s.activity for s in states],dim=0))


def unpack_continuous_state(state,count=None):
    if not isinstance(state,ContinuousConnectomeState): raise ValueError('Explicit continuous state required')
    count = state.activity.shape[0] if count is None else count
    if type(count) is not int or not 1 <= count <= state.activity.shape[0]:
        raise ValueError('Only real state lanes may be unpacked')
    return [ContinuousConnectomeState(state.activity[i:i+1].clone()) for i in range(count)]


def load_full_ports(graph,directory,*,annotations_path=None):
    """Join every annotated VP root to the graph; keep the original1299 DN order."""
    import pandas as pd
    directory = Path(directory)
    annotation = Path(annotations_path) if annotations_path is not None else directory/'annotations/flywire_annotations_v2.1.0.tsv'
    if file_sha256(annotation) != ANNOTATIONS_SHA256:
        raise ValueError('Pinned v2.1.0/v783 annotation bytes required')
    table = pd.read_csv(annotation,sep='\t',dtype=str,keep_default_na=False,usecols=['root_id','super_class'])
    if table['root_id'].duplicated().any():raise ValueError('One anatomical annotation per exact root ID required')
    lookup = {root:i for i,root in enumerate(graph.node_ids)}
    def resolve(label):
        roots = table.loc[table.super_class==label,'root_id'].tolist()
        return tuple(sorted(lookup[root] for root in roots if root in lookup)),len(roots)
    inputs,visual_annotation_rows = resolve('visual_projection')
    annotated_DN,DN_annotation_rows = resolve('descending')
    path = directory/'annotations/selected_ports_1024_v783.json'
    if file_sha256(path) != LEGACY_DN_MANIFEST_SHA256:
        raise ValueError('Original descending-port identity must remain pinned')
    old = json.loads(path.read_text(encoding='utf-8'))
    outputs = tuple(lookup[root] for root in old['output_root_ids'])
    if (outputs != tuple(old['output_indices']) or outputs != annotated_DN or len(outputs) != 1299
            or not inputs or set(inputs)&set(outputs)):
        raise ValueError('Exact full-VP root join and unchanged1299 descending identities required')
    identity = dict(input_root_ids=[graph.node_ids[i] for i in inputs],
        output_root_ids=[graph.node_ids[i] for i in outputs])
    provenance = dict(annotation_sha256=ANNOTATIONS_SHA256,annotation_file=str(annotation.resolve()),
        input_class='visual_projection',output_class='descending',
        visual_annotation_rows=visual_annotation_rows,graph_matched_visual_ports=len(inputs),
        visual_annotation_roots_absent_from_graph=visual_annotation_rows-len(inputs),
        descending_annotation_rows=DN_annotation_rows,graph_matched_descending_ports=len(outputs),
        visual_selection='all root-ID-matched visual_projection in original graph row order;no subsampling',
        old_DN_manifest_sha256=LEGACY_DN_MANIFEST_SHA256,
        ports_identity_sha256=sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest())
    return inputs,outputs,provenance


class ContinuousConnectomeCore(nn.Module):
    def __init__(self,graph,input_indices,output_indices,*,K=8,trainable_brain=True,
            require_full_graph=True,seed=613711,edge_chunk_size=65536,epsilon=1e-6,
            port_provenance=None):
        super().__init__()
        if not isinstance(graph,ConnectomeGraph):raise ValueError('Existing exact directed source graph required')
        if K not in (4,8,16) or type(K) is not int:raise ValueError('Explicit continuous inner updates4/8/16 required')
        if type(trainable_brain) is not bool or type(require_full_graph) is not bool:
            raise ValueError('Explicit trainable/fixed and production/fixture flags required')
        if type(seed) is not int or not 0 <= seed < 2**63 or not 0 < epsilon < .5:
            raise ValueError('Finite initialization seed and positive gain/alpha epsilon required')
        fingerprint = graph_fingerprint(graph)
        if require_full_graph and (graph.n_neurons!=EXPECTED_NEURONS or graph.n_edges!=EXPECTED_EDGES
                or fingerprint!=PINNED_GRAPH_FINGERPRINT):
            raise ValueError('Continuous production requires the same actual full138639/15091983 graph')
        def ports(values):
            values=tuple(values)
            if (not values or any(type(i) not in (int,np.int32,np.int64) for i in values)
                    or len(set(values))!=len(values) or min(values)<0 or max(values)>=graph.n_neurons):
                raise ValueError('Unique in-graph integer sensory/descending ports required')
            return tuple(int(i) for i in values)
        inputs,outputs=ports(input_indices),ports(output_indices)
        if set(inputs)&set(outputs):raise ValueError('VP input and DN output must remain disjoint')
        if require_full_graph and (port_provenance is None
                or port_provenance.get('annotation_sha256')!=ANNOTATIONS_SHA256
                or port_provenance.get('graph_matched_visual_ports')!=len(inputs)
                or len(outputs)!=1299):
            raise ValueError('Production ports require the verified all-VP root-ID join')
        self.graph,self.K,self.epsilon,self.initialization_seed=graph,K,float(epsilon),seed
        self.n_neurons,self.n_edges=graph.n_neurons,graph.n_edges
        self.register_buffer('input_indices',torch.tensor(inputs,dtype=torch.long),persistent=False)
        self.register_buffer('output_indices',torch.tensor(outputs,dtype=torch.long),persistent=False)
        # Degree sums use the ORIGINAL absolute prior, never learned gains.
        absolute=np.abs(graph.signed_weights.astype(np.float64))
        incoming=np.bincount(graph.post,weights=absolute,minlength=graph.n_neurons)
        outgoing=np.bincount(graph.pre,weights=absolute,minlength=graph.n_neurons)
        denominator=np.sqrt(incoming[graph.post]*outgoing[graph.pre])
        normalized=np.divide(graph.signed_weights,denominator,
            out=np.zeros(graph.n_edges,dtype=np.float64),where=denominator>0).astype(np.float32)
        if (not np.isfinite(normalized).all()
                or not np.array_equal(np.sign(normalized),np.sign(graph.signed_weights))):
            raise ValueError('Degree normalization must retain every original sign/nonzero edge')
        self.register_buffer('normalized_signed_prior',torch.from_numpy(normalized),persistent=False)
        self.sparse=TrainableDirectedSparse(graph.n_neurons,graph.pre,graph.post,edge_chunk_size=edge_chunk_size)
        initial_positive=math.log(math.expm1(1.-self.epsilon))
        self.edge_gain_logits=nn.Parameter(torch.full((graph.n_edges,),initial_positive,dtype=torch.float32))
        self.alpha_logits=nn.Parameter(torch.zeros(graph.n_neurons,dtype=torch.float32))
        self.neuron_bias=nn.Parameter(torch.zeros(graph.n_neurons,dtype=torch.float32))
        self.message_gain_logit=nn.Parameter(torch.tensor(initial_positive,dtype=torch.float32))
        # Input initialization is shared exactly by fixed/trainable controls.
        with torch.random.fork_rng(devices=[]):
            self.input_encoder=nn.Linear(384,len(inputs),bias=True)
            generator=torch.Generator(device='cpu').manual_seed(seed)
            bound=1./math.sqrt(384)
            nn.init.uniform_(self.input_encoder.weight,-bound,bound,generator=generator)
            nn.init.zeros_(self.input_encoder.bias)
        self.graph_provenance=dict(copy_of_original_source=True,graph_fingerprint=fingerprint,
            source=dict(graph.provenance),n_neurons=graph.n_neurons,n_edges=graph.n_edges,
            positive_edges=int((graph.signed_weights>0).sum()),negative_edges=int((graph.signed_weights<0).sum()),
            original_zero_weight_entries_retained=int((graph.signed_weights==0).sum()),
            all_source_edges_retained=True,source_edge_order_retained=True,
            normalization='signed_prior[e]/sqrt(sum_abs_in[post[e]]*sum_abs_out[pre[e]])',
            degree_statistics_dtype='float64 CPU;normalized signed prior float32',
            degree_normalization_uses_original_weights=True,port_provenance=port_provenance or dict(synthetic_fixture=True),
            require_full_graph=require_full_graph)
        self.set_trainable_brain(trainable_brain)

    @classmethod
    def from_full_connectome(cls,directory,*,annotations_path=None,K=8,trainable_brain=True,
            device='cpu',seed=613711,**kwargs):
        graph=load_connectome(directory,max_neurons=None,verify=True)
        inputs,outputs,provenance=load_full_ports(graph,directory,annotations_path=annotations_path)
        result=cls(graph,inputs,outputs,K=K,trainable_brain=trainable_brain,require_full_graph=True,
            seed=seed,port_provenance=provenance,**kwargs)
        return result.to(device)

    @property
    def device(self):return self.input_encoder.weight.device
    def internal_parameters(self):
        return (self.edge_gain_logits,self.alpha_logits,self.neuron_bias,self.message_gain_logit)
    def set_trainable_brain(self,enabled):
        if type(enabled) is not bool:raise ValueError('Explicit internal-learning switch required')
        self.trainable_brain=enabled
        for p in self.internal_parameters():p.requires_grad_(enabled);p.grad=None
        for p in self.input_encoder.parameters():p.requires_grad_(True)
        return self
    def positive_edge_gains(self):return self.epsilon+F.softplus(self.edge_gain_logits)
    def effective_edge_values(self):return self.normalized_signed_prior*self.positive_edge_gains()
    def alpha(self):return self.epsilon+(1.-2.*self.epsilon)*torch.sigmoid(self.alpha_logits)
    def message_gain(self):return self.epsilon+F.softplus(self.message_gain_logit)

    def initial_state(self,batch,*,device=None):
        if type(batch) is not int or batch < 1:raise ValueError('Positive actual batch required')
        target=self.device if device is None else torch.device(device)
        return ContinuousConnectomeState(torch.zeros(batch,self.n_neurons,dtype=torch.float32,device=target))

    def forward(self,encoded,state):
        if (not isinstance(state,ContinuousConnectomeState) or encoded.ndim!=2 or encoded.shape[1]!=384
                or encoded.dtype!=torch.float32 or encoded.device!=self.device
                or state.activity.shape!=(len(encoded),self.n_neurons) or state.activity.device!=self.device):
            raise ValueError('Same-device encoded384 and complete FP32 activity state required')
        activity=state.activity
        drive=torch.zeros_like(activity).index_add(1,self.input_indices,self.input_encoder(encoded))
        values,alpha,gain=self.effective_edge_values(),self.alpha(),self.message_gain()
        for _ in range(self.K):
            message=self.sparse(activity,values)
            activity=(1.-alpha)*activity+alpha*torch.tanh(gain*message+drive+self.neuron_bias)
        following=ContinuousConnectomeState(activity)
        return activity.index_select(1,self.output_indices),following

    def config(self):
        return dict(schema=SCHEMA,n_neurons=self.n_neurons,n_edges=self.n_edges,encoded_dim=384,
            input_ports=len(self.input_indices),output_ports=len(self.output_indices),K=self.K,
            scalar_states_per_node=1,state='continuous FP32 activity[B,N]',
            equation="h'=(1-alpha)*h+alpha*tanh(message_gain*(W_eff@h)+port_drive+b)",
            graph_fingerprint=self.graph_provenance['graph_fingerprint'],
            ports_identity_sha256=self.graph_provenance['port_provenance'].get('ports_identity_sha256'),
            normalization=self.graph_provenance['normalization'],
            edge_gains='epsilon+softplus(edge_gain_logits);initial1',epsilon=self.epsilon,
            alpha='epsilon+(1-2epsilon)*sigmoid(alpha_logits);initial.5',
            neuron_bias_initial=0.,message_gain='scalar epsilon+softplus(logit);initial1',
            input_initialization='private CPU seed;uniform +/-1/sqrt384;bias0',seed=self.initialization_seed,
            trainable_brain=self.trainable_brain,input_encoder_trainable=True,
            graph_signs_topology_fixed=True,new_or_deleted_edges=0,
            sparse=self.sparse.config(),LIF_or_spikes=False,neural_RNG=False,
            reset='genuine new native game only;caller-owned;not stage/Miss',
            checkpoint='named parameters only;fixed graph/ports reconstructed and verified',
            old_LIF_checkpoint_compatible=False)
