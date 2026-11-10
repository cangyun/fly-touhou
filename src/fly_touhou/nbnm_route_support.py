"""Count physical exits, not keys that push against the same wall."""
import numpy as np


def physical_exit_support(starts, velocities, safe, *, duration=12, cell=8.):
    starts=np.asarray(starts);velocities=np.asarray(velocities);safe=np.asarray(safe,bool)
    ends=np.clip(starts[:,None,:]+velocities[None,:,:]*duration,[-184.,32.],[184.,432.])
    delta=ends-starts[:,None,:]
    # Focus and unfocused idle have the same physical exit. At a wall, many
    # outward keys collapse to that exit too; count it once, never as escape.
    denominator=len(np.unique(velocities,axis=0))
    fractions=np.array([len(np.unique(np.rint(ends[i,mask]/cell),axis=0))/denominator
        if mask.any() else 0. for i,mask in enumerate(safe)],np.float32)
    horizontal=(safe&(delta[:,:,0]<-1)).any(-1)&(safe&(delta[:,:,0]>1)).any(-1)
    vertical=(safe&(delta[:,:,1]<-1)).any(-1)&(safe&(delta[:,:,1]>1)).any(-1)
    opposed=(horizontal.astype(np.float32)+vertical.astype(np.float32))/2
    return fractions,opposed
