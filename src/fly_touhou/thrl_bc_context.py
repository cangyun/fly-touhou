"""Current-parameter pre-action GRU states replayed from real episode starts.

Caches are detached truncated-BPTT initial states, not additional observations.
Mirror contexts replay the entire past in the same orientation as its chunk.
"""
from __future__ import annotations


def recurrent_context_cache(model,fetch,worlds,frames,*,mirrored=False,chunk_size=128):
    import torch
    from .thrl_augmentation import horizontal_mirror
    if not model.is_recurrent:raise ValueError('Recurrent context requires GRU')
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    result=torch.zeros(worlds*frames,model.gru_hidden_size,device=device,dtype=dtype)
    was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            for world in range(worlds):
                hidden=torch.zeros(1,1,model.gru_hidden_size,device=device,dtype=dtype)
                for first in range(0,frames,chunk_size):
                    stop=min(frames,first+chunk_size);batch=fetch(world,first,stop)
                    f,m,o,y=(batch[k] for k in ('features','maps','objects','targets'))
                    if mirrored:
                        f,m,o,_=horizontal_mirror(f[None],m[None],o[None],y[None],[True])
                        f,m,o=f[0],m[0],o[0]
                    encoded=model._encode(f,m,o)
                    dones=batch['dones'].bool()
                    boundaries=(torch.nonzero(dones,as_tuple=True)[0]+1).tolist()
                    if not boundaries or boundaries[-1]!=len(encoded):boundaries.append(len(encoded))
                    begin=0
                    for end in boundaries:
                        offset=world*frames+first+begin
                        result[offset]=hidden[0,0]
                        outputs,hidden=model.gru(encoded[begin:end,None],hidden)
                        if end-begin>1:result[offset+1:offset+end-begin]=outputs[:-1,0]
                        if dones[end-1]:hidden=torch.zeros_like(hidden)
                        begin=end
    finally:model.train(was_training)
    return result.detach()
