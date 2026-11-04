"""Read the retained training tape without loading its10GB brain archive."""
import os
from pathlib import Path


def inspect(output):
    import psutil
    import numpy as np
    from apply_compact_live_wall import frames
    from fly_touhou.compact_native_contact_targets import native_exposure_mask
    from fly_touhou.thrl_durable_records import write_json
    assert os.getpid()==22972 and abs(psutil.Process().create_time()-1793563102.6926827)<.01
    choices=[]
    for frame in frames():
        if Path(frame.f_code.co_filename).name!='train_compact_story_ppo.py':continue
        for name,value in frame.f_locals.items():
            if isinstance(value,dict) and 'recorded_observed_contacts' in value and 'features' in value:
                choices.append((name,value))
    name,data=choices[0]
    def array(key):return data[key].detach().cpu().numpy()
    f=array('features');actions=array('actions');outcomes=array('outcomes');facts=array('recorded_observed_contacts')
    valid=array('valid_mask') if 'valid_mask' in data else np.ones_like(actions,dtype=bool)
    dt=array('native_action_frames');normal=array('unprotected_mask');exposed=native_exposure_mask(f,normal)
    expiry=valid&exposed&~normal&(actions>=0)&(actions<18)
    recoverable=expiry&(outcomes<0)&(((dt>0)&(dt<=4)&facts)|((dt==4)&~facts))
    stages=array('recorded_native_stage')
    result=dict(source='retained live TRAINING tape',local_name=name,shape=list(actions.shape),
        valid_rows=int(valid.sum()),expiry_rows=int(expiry.sum()),recoverable=int(recoverable.sum()),
        recoverable_contacts=int((recoverable&facts).sum()),recoverable_clears=int((recoverable&~facts).sum()),
        by_stage={str(s):dict(rows=int((recoverable&(stages==s)).sum()),
            contacts=int((recoverable&facts&(stages==s)).sum())) for s in range(1,7)},
        raw_tape_modified=False,parameters_modified=False)
    write_json(Path(output),result)
    return result
