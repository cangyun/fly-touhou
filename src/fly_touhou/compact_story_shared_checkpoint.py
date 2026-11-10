"""Recompute each candidate chunk once from its shared pre-action brain state.

Equivalent recorded-action training path. Candidate outputs and the selected
following state are unchanged;only saved activations/recomputation differ.
"""
import torch
from torch.utils.checkpoint import checkpoint
from .spiking_engineered_control import SpikingState
from .compact_story_model import story_candidates


def committed_step(model,features,state,actions):
    width=model.core.sensory_width
    n=len(features);cue=(model.candidate_features(features) if hasattr(model,'candidate_features') else story_candidates(features)).reshape(-1,width)
    chosen=torch.arange(n,device=features.device)*18+torch.where(actions==18,0,actions)
    following=model.initial_state(n);DNs=[]
    for first in range(0,len(cue),128):
        count=min(128,len(cue)-first)
        def call(x,chosen,*parts,first=first,count=count):
            # Store only the unreplicated state at the checkpoint boundary.
            # Repeated candidate state exists only inside the chunk forward.
            worlds=torch.arange(first,first+count,device=x.device)//18
            before=SpikingState(*(v.index_select(0,worlds) for v in parts))
            if count<128 and model.pad_candidate_batch:
                zeros=model.initial_state(128-count)
                x=torch.cat((x,x.new_zeros(128-count,width)))
                before=SpikingState(*(torch.cat((v,z)) for v,z in zip(before.tensors(),zeros.tensors())))
            DN,after=model.core(x,before)
            local=(chosen-first).clamp(0,count-1)
            return DN[:count],*(v.index_select(0,local) for v in after.tensors())
        result=checkpoint(call,cue[first:first+count],chosen,*state.tensors(),use_reentrant=False)
        DN,*parts=result;DNs.append(DN)
        selected=((chosen>=first)&(chosen<first+count))[:,None]
        following=SpikingState(*(torch.where(selected,new,old) for new,old in zip(parts,following.tensors())))
    return model.heads(torch.cat(DNs),features),following


def sequence(model,features,starts,state,actions,*,reset_states=None):
    records=[]
    for t in range(len(features)):
        state=state.reset(starts[t])
        if reset_states:
            from .nbnm_prefix import apply_resets
            state=apply_resets(state,reset_states.get(t))
        out,state=committed_step(model,features[t],state,actions[t]);records.append(out)
    return dict(logits=torch.stack([r['logits'] for r in records]),
        values=torch.stack([r['values'] for r in records]),
        horizon_scores=torch.stack([r['horizon_scores'] for r in records]),state=state)
