"""FP32 state-local caches and full-episode recurrent contexts for actor BC.

No native process or policy optimization is performed here. A cache is bound
to unchanged input/risk/critic tensors; only the temporal policy may change.
"""
from __future__ import annotations

import time
import numpy as np

from .thrl_cached_actor_fit import TRAINABLE_PREFIXES


class TeacherLatentCache:
    def __init__(self, torch, model, dataset, *, on_progress, check_stop, chunk_rows=32):
        self.torch, self.model, self.dataset = torch, model, dataset
        self.device = next(model.parameters()).device
        self.progress, self.stop = on_progress, check_stop
        self._last = 0.
        self.frozen = {name:t.detach().cpu().clone() for name,t in model.state_dict().items()
                       if not name.startswith(TRAINABLE_PREFIXES)}
        self.versions = self._versions()
        if any(p.requires_grad for name,p in model.named_parameters()
               if not name.startswith(TRAINABLE_PREFIXES)):
            raise ValueError('Freeze input encoders, critics and fitted risk before creating caches')
        n, width, hidden = dataset.rows, model.gru.input_size, model.gru_hidden_size
        self.estimated_bytes = n*(4*(width+25+18+hidden)+8+2)
        if self.device.type == 'cuda' and torch.cuda.mem_get_info()[0] < self.estimated_bytes+3*1024**3:
            raise RuntimeError('Insufficient CUDA space for exact FP32 caches plus3GiB working reserve')
        self.encoded = torch.empty((n,width),dtype=torch.float32,device=self.device)
        self.features = torch.empty((n,25),dtype=torch.float32,device=self.device)
        self.risk = torch.empty((n,18),dtype=torch.float32,device=self.device)
        self.actions = torch.empty(n,dtype=torch.int64,device=self.device)
        self.dones = torch.empty(n,dtype=torch.bool,device=self.device)
        self.legal = torch.empty(n,dtype=torch.bool,device=self.device)
        self.contexts = torch.empty((n,hidden),dtype=torch.float32,device=self.device)
        self.legal_cpu = np.empty(n,dtype=np.bool_)
        self.stats = dict(rows=n, invalid_inventory_labels=0, action_counts=[0]*19,stage_counts=[0]*7)
        started=time.monotonic();completed=0
        with torch.no_grad():
            for i,record in enumerate(dataset.records):
                for start,stop,data,legal in dataset.iter_chunks(i,chunk_rows=chunk_rows):
                    check_stop();lo,hi=record['offset']+start,record['offset']+stop
                    obs={name:torch.from_numpy(data[name]).to(self.device) for name in ('features','maps','objects')}
                    self.encoded[lo:hi].copy_(model._encode(obs['features'],obs['maps'],obs['objects']))
                    self.risk[lo:hi].copy_(model.precompute_frozen_risk(obs['features'],obs['objects']))
                    self.features[lo:hi].copy_(obs['features'])
                    self.actions[lo:hi].copy_(torch.from_numpy(data['actions']).to(self.device))
                    self.dones[lo:hi].copy_(torch.from_numpy(data['dones']).to(self.device))
                    self.legal[lo:hi].copy_(torch.from_numpy(legal).to(self.device))
                    self.legal_cpu[lo:hi]=legal
                    self.stats['invalid_inventory_labels']+=int((~legal).sum())
                    self.stats['action_counts']=(np.array(self.stats['action_counts'])+
                        np.bincount(data['actions'][legal],minlength=19)).tolist()
                    self.stats['stage_counts']=(np.array(self.stats['stage_counts'])+
                        np.bincount(data['stage'],minlength=7)).tolist()
                    completed+=stop-start
                    self._publish('precomputing_frozen_encoders',completed=completed,total=n,
                                  wall_seconds=time.monotonic()-started)
        if completed!=n:
            raise ValueError('Every original row must enter the detached cache once')
        dataset.close_caches()
        self.assert_valid()
        self.stats.update(precomputation_seconds=time.monotonic()-started,estimated_cache_bytes=self.estimated_bytes,
                          original_FP32=True,quantized=False,encoder_frozen=True,risk_frozen=True)
        self._publish('precomputing_frozen_encoders',force=True,completed=n,total=n,**self.stats)

    def _versions(self):
        return {name:t._version for name,t in self.model.state_dict(keep_vars=True).items()
                if not name.startswith(TRAINABLE_PREFIXES)}

    def assert_valid(self):
        if self._versions()!=self.versions:
            raise ValueError('Cached input/risk/critic tensors changed;recompute before further updates')

    def rebind_after_best_restore(self):
        state=self.model.state_dict()
        if any(not self.torch.equal(state[name].detach().cpu(),value) for name,value in self.frozen.items()):
            raise ValueError('Restored policy changed a frozen cache source')
        self.versions=self._versions()

    def _publish(self,phase,*,force=False,**values):
        now=time.monotonic()
        if force or now-self._last>=5:
            self.progress(dict(phase=phase,**values));self._last=now

    def refresh_contexts(self):
        """Replay every actual past observation; remove a world after its end.

        Worlds are batched at the same local decision index. There is no fake
        padding observation, segment reset, replay landing, or zeroed chunk.
        """
        torch=self.torch;self.assert_valid();started=time.monotonic()
        modes=[(module,module.training) for module in self.model.modules()]
        self.model.eval()
        lengths=[record['rows'] for record in self.dataset.records]
        offsets=[record['offset'] for record in self.dataset.records]
        hidden=torch.zeros(1,len(lengths),self.model.gru_hidden_size,device=self.device)
        active=list(range(len(lengths)));filled=0
        try:
            with torch.no_grad():
                for t in range(max(lengths)):
                    if t%128==0:self.stop()
                    active=[i for i in active if t<lengths[i]]
                    index=torch.tensor([offsets[i]+t for i in active],device=self.device,dtype=torch.int64)
                    previous=hidden[:,active].contiguous()
                    self.contexts[index]=previous[0]
                    _,following=self.model.gru(self.encoded[index].unsqueeze(0),previous)
                    hidden[:,active]=following
                    filled+=len(active)
                    if t%128==0:
                        self._publish('refreshing_full_trajectory_context',completed=filled,total=self.dataset.rows,
                                      wall_seconds=time.monotonic()-started)
        finally:
            for module,mode in modes:module.training=mode
        if filled!=self.dataset.rows:
            raise ValueError('Full recurrent context did not cover the original archive')
        self.context_seconds=time.monotonic()-started
        return self.context_seconds

    def valid_descriptors(self,indices,*,length):
        descriptors=self.dataset.descriptors(indices,length=length)
        # No runtime loss/KL batch rejection. Discard only blocks that have no
        # inventory-legal imitation target; retain their real context/history.
        selected=[(i,start) for i,start in descriptors if self.legal_cpu[
            self.dataset.records[i]['offset']+start:self.dataset.records[i]['offset']+start+length].any()]
        if not selected:
            raise ValueError('This trajectory split has no legal teacher action labels')
        return selected,len(descriptors)-len(selected)

    def batch(self,descriptors,*,length):
        torch=self.torch
        first=torch.tensor([self.dataset.records[i]['offset']+start for i,start in descriptors],
                           device=self.device,dtype=torch.int64)
        positions=first[:,None]+torch.arange(length,device=self.device)[None]
        return dict(encoded=self.encoded[positions],features=self.features[positions],risk=self.risk[positions],
                    initial_hidden=self.contexts[first][None].contiguous(),dones=self.dones[positions],
                    actions=self.actions[positions].reshape(-1),legal=self.legal[positions].reshape(-1))
