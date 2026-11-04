"""One passive intermediate brain capsule in the current frozen evaluation."""
from pathlib import Path


def install():
    import numpy as np
    import torch
    from apply_compact_live_wall import frames
    frame=next(f for f in frames() if f.f_code.co_name=='evaluate'
        and Path(f.f_code.co_filename).name=='compact_full_story.py')
    local=frame.f_locals;model=local['model'];output=local['output']
    if local['episodes']!=1:raise ValueError('This passive capsule records the single seed3 game')
    original=model.infer_candidates;infos=local['infos'];opening_seed=local['plans'][0]
    destination=output/'stage5-prefix';destination.mkdir(parents=True,exist_ok=True)
    def observed(features,states,**kwargs):
        if len(features)==1 and round(float(features[0,111])*6)==5:
            torch.save(dict(schema='frozen-SNN-stage-entry-brain-v2',
                state=[v.detach().cpu().clone() for v in states[0].tensors()],
                features=np.array(features[0],copy=True),opening_seed=opening_seed,
                context={k:infos[0].get(k) for k in ('stage','stage_frame','episode_frames','lives','bombs','misses','bombs_used')},
                frozen_decoding=getattr(model,'frozen_decoding',None),optimizer_updates=0),
                destination/'brain-world-0.pt')
            model.infer_candidates=original
        return original(features,states,**kwargs)
    model.infer_candidates=observed
    return dict(status='armed',source='Actual frozen prefix brain before its first stage5 decision',
        output=str(destination/'brain-world-0.pt'),actions_changed=False,optimizer_updates=0)
