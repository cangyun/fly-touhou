"""Retrospective movement examples from recorded training-game transitions."""
import numpy as np


def recover_recorded_contacts(data):
    """Recover formerly omitted protected transitions from paired observations.

    No geometry prediction or evaluation game is needed. The last unpaired
    row stays unchanged; frame gaps and episode boundaries are not bridged.
    """
    old=np.asarray(data['outcome']);out=old.copy();f=data['features']
    elapsed=data['native_action_frames'];frame=data['before_episode_frames']
    paired=(np.diff(frame)==elapsed[:-1])&(elapsed[:-1]>0)
    paired &= ~np.asarray(data['episode_starts'][1:],bool)
    selected=paired&(old[:-1]<0)&(f[:-1,14]>.5)&(data['actions'][:-1]<18)
    hit=(np.diff(data['misses_before'])>0)|((f[1:,17]>.5)&(f[:-1,17]<.5))
    rows=np.flatnonzero(selected);out[rows]=hit[rows].astype(out.dtype)
    return out,dict(recovered_rows=len(rows),recovered_contacts=int(hit[rows].sum()),
        source='Contiguous recorded next-state contact window and miss count;no inferred alternative outcomes')


def demonstrated_protected_continuation(data, *, horizon_after_exit=96):
    """Require a clean actual trajectory through shield expiry and96f beyond.

    A currently shielded action gets no positive risk label. This mask supplies
    only weak imitation of the teacher's actual movement during repositioning.
    """
    f=data['features'];n=len(f);indices=np.arange(n)
    normal=np.asarray(data['unprotected_mask'],bool)
    exit_row=np.minimum.accumulate(np.where(normal,indices,n)[::-1])[::-1]
    frame=data['before_episode_frames'];last=np.minimum(exit_row,n-1)
    stop=np.searchsorted(frame,frame[last]+horizon_after_exit,side='left')
    bad=(data['outcome']!=0)|(data['actions']>=18)|(data['native_action_frames']!=4)
    prefix=np.concatenate(([0],np.cumsum(bad,dtype=np.int64)))
    clear=prefix[np.minimum(stop,n)]==prefix[indices]
    protected=(f[:,14]>.5)&(f[:,13]>0)&~normal
    valid=protected&(exit_row<n)&(stop<n)&clear
    valid &= data['misses_before'][np.minimum(stop,n-1)]==data['misses_before']
    return valid,stop
