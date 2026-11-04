"""Lossless bounded observation history, refreshed with CURRENT parameters.

Packed observations stay in RAM. No FP32 map archive on C and no old hidden
state is recycled after an Adam update. Finite burn-in is explicitly an
approximation to complete-prefix recurrent state, not claimed identical.
"""
from collections import deque
import io
import zlib
import numpy as np


def pack_observations(observations):
    if not observations:
        return None
    stream=io.BytesIO()
    np.savez(stream,**{key:np.stack([r[key] for r in observations]) for key in observations[0]})
    return zlib.compress(stream.getvalue(),level=1)


def unpack_observations(payload):
    if payload is None:
        return []
    with np.load(io.BytesIO(zlib.decompress(payload)),allow_pickle=False) as arrays:
        count=len(arrays[arrays.files[0]])
        if any(arrays[k].dtype!=np.float32 or len(arrays[k])!=count for k in arrays.files):
            raise ValueError('Invalid lossless recurrent history')
        return [{key:arrays[key][i].copy() for key in arrays.files} for i in range(count)]


class PolicyHistory:
    def __init__(self,count,length,*,enabled=True,storage='compressed-v1'):
        if not 1<=length<=128:
            raise ValueError('Bounded recurrent history must be 1..128 decisions')
        self.rows=[deque(maxlen=length) for _ in range(count)]
        if storage not in ('compressed-v1','raw-v1'):raise ValueError('Unknown lossless recurrent history storage')
        self.storage=storage
        self.enabled=bool(enabled)
        self.refreshes=0

    def reset(self,index,payload=None):
        self.rows[index].clear()
        if not self.enabled:return
        for obs in unpack_observations(payload):
            self.append(index,obs)

    def append(self,index,observation):
        if not self.enabled:return
        if self.storage=='raw-v1':
            if any(value.dtype!=np.float32 for value in observation.values()):
                raise ValueError('Raw recurrent history must preserve float32 observations')
            self.rows[index].append({key:value.copy() for key,value in observation.items()})
        else:self.rows[index].append(pack_observations([observation]))

    def refresh(self,torch,model,indices=None):
        indices=list(range(len(self.rows))) if indices is None else list(indices)
        device=next(model.parameters()).device
        hidden=torch.zeros(1,len(indices),model.gru_hidden_size,device=device)
        if not self.enabled or not getattr(model,'is_recurrent',True):
            model.eval()
            return hidden
        histories=[list(self.rows[i]) for i in indices]
        def decode(payload):return payload if isinstance(payload,dict) else unpack_observations(payload)[0]
        # Encode small ready batches together; do not wait for future game
        # frames or alter actual decision intervals. Unequal history is padded
        # on the left by no operation, never by fake observations.
        model.eval()
        from .thrl_inputs import forward_observation
        with torch.no_grad():
            # Frame encoders are independent. Batch them across worlds AND
            # history time, retaining only small embeddings on the GPU.
            # Decode a bounded chunk instead of materializing gigabytes of maps.
            encoded=None
            if hasattr(model,'_encode') and hasattr(model,'gru'):
                encoded=[[] for _ in histories]
                flat=[(j,payload) for j,row in enumerate(histories) for payload in row]
                for lo in range(0,len(flat),64):
                    chunk=flat[lo:lo+64]
                    observations=[decode(payload) for _,payload in chunk]
                    batch={key:torch.as_tensor(np.stack([r[key] for r in observations]),device=device)
                           for key in observations[0]}
                    values=model._encode(batch['features'],batch['maps'],batch.get('objects'))
                    for k,(j,_) in enumerate(chunk):encoded[j].append(values[k])
            for t in range(max((len(h) for h in histories),default=0)):
                active=[j for j,h in enumerate(histories) if t<len(h)]
                for lo in range(0,len(active),32):
                    js=active[lo:lo+32]
                    if encoded is not None:
                        _,state=model.gru(torch.stack([encoded[j][t] for j in js])[None],hidden[:,js])
                    else:
                        rows=[decode(histories[j][t]) for j in js]
                        observations={key:np.stack([r[key] for r in rows]) for key in rows[0]}
                        _,_,state=forward_observation(torch,model,observations,hidden[:,js])
                    hidden[:,js]=state
        self.refreshes+=1
        return hidden

    def summary(self):
        return dict(burn_in_max=self.rows[0].maxlen if self.rows and self.enabled else 0,
                    retained_observations=sum(map(len,self.rows)),
                    compressed_bytes=sum(len(p) for row in self.rows for p in row if isinstance(p,bytes)),
                    raw_bytes=sum(sum(a.nbytes for a in p.values()) for row in self.rows for p in row if isinstance(p,dict)),
                    storage=self.storage,
                    current_parameter_refreshes=self.refreshes,
                    complete_prefix_equivalence_claimed=False,history_disk_files=0,
                    recurrent_enabled=self.enabled)
