"""Recompute each candidate chunk once from its shared pre-action brain state.

Equivalent recorded-action training path. Candidate outputs and the selected
following state are unchanged;only saved activations/recomputation differ.
"""
import torch
from torch.utils.checkpoint import checkpoint
from .spiking_engineered_control import SpikingState
from .compact_story_model import story_candidates


def committed_step(model,features,state,actions,*,core_options=None):
    core_options = core_options or {}
    width=model.core.sensory_width
    n=len(features);cue=(model.candidate_features(features) if hasattr(model,'candidate_features') else story_candidates(features)).reshape(-1,width)
    chosen=torch.arange(n,device=features.device)*18+torch.where(actions==18,0,actions)
    initial = None
    ordered = core_options.get('ordered_training')
    if ordered is not None and getattr(model,'reuse_initial_message',False):
        sparse = model.core.sparse
        with torch.no_grad():
            spikes = state.spikes
            if len(cue) % 128 and model.pad_candidate_batch:
                spikes = torch.cat((spikes, spikes.new_zeros(1, spikes.shape[1])))
            # Before applying directional input, all candidates from this world
            # have identical spikes. Keep the existing per-candidate backward.
            initial = sparse._multiply(spikes,None,transpose=False,ordered=ordered[0],
                compact=sparse.compact_training)
    retained = getattr(model,'retained_message_steps',0)
    following=model.initial_state(n);DNs=[]
    for first in range(0,len(cue),128):
        count=min(128,len(cue)-first)
        worlds=torch.arange(first,first+count,device=features.device)//18
        chunk_options = dict(core_options)
        if retained:
            chunk_options['message_cache']={tick:None for tick in range(max(1,model.core.K-retained),model.core.K)}

        def call(x,chosen,*parts,worlds=worlds,first=first,count=count,options=chunk_options):
            # Store only the unreplicated state at the checkpoint boundary.
            # Repeated candidate state exists only inside the chunk forward.
            before=SpikingState(*(v.index_select(0,worlds) for v in parts))
            message=initial.index_select(0,worlds) if initial is not None else None
            if count<128 and model.pad_candidate_batch:
                zeros=model.initial_state(128-count)
                x=torch.cat((x,x.new_zeros(128-count,width)))
                before=SpikingState(*(torch.cat((v,z)) for v,z in zip(before.tensors(),zeros.tensors())))
                if message is not None:
                    message=torch.cat((message,initial[-1:].expand(128-count,-1)))
            if message is not None:options=dict(options,initial_message=message)
            DN,after=model.core(x,before,**options)
            local=(chosen-first).clamp(0,count-1)
            return DN[:count],*(v.index_select(0,local) for v in after.tensors())
        result=checkpoint(call,cue[first:first+count],chosen,*state.tensors(),use_reentrant=False)
        DN,*parts=result;DNs.append(DN)
        selected=((chosen>=first)&(chosen<first+count))[:,None]
        following=SpikingState(*(torch.where(selected,new,old) for new,old in zip(parts,following.tensors())))
    return model.heads(torch.cat(DNs),features),following


def sequence(model,features,starts,state,actions,*,reset_states=None):
    records=[]
    core_options = {}
    op = getattr(model.core, 'sparse', None)
    if (torch.is_grad_enabled() and features.is_cuda and getattr(op, 'reuse_sequence_weights', False)
            and op.deterministic_spmm and op.reuse_ordered_weights):
        # Every checkpoint in this sequence sees the same parameters. Keep
        # only the detached CSR layouts shared. Each core retains its own
        # edge-value autograd graph and accumulates gradients in the same order.
        with torch.no_grad():
            core_options['ordered_training'] = op.ordered_weights(model.core.effective_edge_values())
    for t in range(len(features)):
        state=state.reset(starts[t])
        if reset_states:
            from .nbnm_prefix import apply_resets
            state=apply_resets(state,reset_states.get(t))
        out,state=committed_step(model,features[t],state,actions[t],core_options=core_options);records.append(out)
    result = dict(logits=torch.stack([r['logits'] for r in records]),
        values=torch.stack([r['values'] for r in records]),
        horizon_scores=torch.stack([r['horizon_scores'] for r in records]),state=state)
    if 'warning48_logits' in records[0]:
        result['warning48_logits'] = torch.stack([r['warning48_logits'] for r in records])
    if 'corridor_logits' in records[0]:
        result['corridor_logits'] = torch.stack([r['corridor_logits'] for r in records])
    if 'corridor_witness_logits' in records[0]:
        result['corridor_witness_logits']=torch.stack([r['corridor_witness_logits'] for r in records])
    if 'corridor_course_logits' in records[0]:
        result['corridor_course_logits']=torch.stack([r['corridor_course_logits'] for r in records])
    return result
