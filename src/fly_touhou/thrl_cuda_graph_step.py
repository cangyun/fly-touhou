"""Unregistered inference-only CUDA graph prototype for the existing B1 policy.

No game/collector integration. Weights, optimizer and sequence training remain
outside this object. Outputs are caller-owned; padding never joins worlds.
"""
from __future__ import annotations
import torch

SCHEMA='B1-frozen-shape-CUDA-step-prototype-v1'


class CapturedEntityStep:
    def __init__(self,model,*,maximum_batch=30):
        from .thrl_pair_policy import EntityRiskPolicy
        if not isinstance(model,EntityRiskPolicy) or not model.frozen_readout:
            raise ValueError('Explicit existing frozen-readout B1 model required')
        if any(p.requires_grad for p in model.risk.parameters()):
            raise ValueError('The declared risk readout must actually be frozen')
        if model.training:raise ValueError('Capture prototype requires an explicitly frozen eval mode')
        if type(maximum_batch) is not int or not 1<=maximum_batch<=30:
            raise ValueError('Prototype ceiling is30; no native worlds are started')
        p=next(model.parameters())
        if p.device.type!='cuda' or p.dtype!=torch.float32:
            raise ValueError('CUDA FP32 model required')
        if any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):
            raise ValueError('Only the existing zero-dropout policy is qualified')
        self.model=model;self.device=p.device;self.maximum_batch=maximum_batch
        self.buckets=sorted({min(maximum_batch,b) for b in (1,2,4,8,16,32)})
        self.graphs={};self.calls=0
        self.signature=self._signature();self.settings=self._settings()

    def _signature(self):
        return tuple((id(p),p.data_ptr(),tuple(p.shape),p.dtype,p.device)
            for p in (*self.model.parameters(),*self.model.buffers()))

    def _settings(self):
        m=self.model
        return (m.beta,m.buffer,m.risk_width,m.input_schema,m.frozen_readout,
            tuple(child.training for child in m.modules()),tuple(p.requires_grad for p in m.risk.parameters()))

    def _capture(self,batch):
        d=self.device
        f=torch.zeros(batch,25,device=d);maps=torch.zeros(batch,16,92,96,device=d)
        objects=torch.zeros(batch,96,21,device=d)
        h=torch.zeros(1,batch,self.model.gru_hidden_size,device=d)
        stream=torch.cuda.Stream(device=d)
        stream.wait_stream(torch.cuda.current_stream(d))
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):self.model.forward_step(f,maps,h,objects=objects)
        torch.cuda.current_stream(d).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(graph,stream=stream):
            # Redundant Python .all() guards cannot run inside CUDA capture.
            # The equivalent output guard is a captured tensor and is checked
            # after every replay, before any result is returned to a caller.
            z,v,following=self.model.forward_step(f,maps,h,objects=objects,check_finite=False)
            valid=(torch.isfinite(z).logical_or(torch.isneginf(z)).all()
                &torch.isfinite(z).any(-1).all()&torch.isfinite(v).all()&torch.isfinite(following).all())
        row=dict(graph=graph,stream=stream,features=f,maps=maps,objects=objects,hidden=h,
            logits=z,values=v,following=following,finite=valid)
        self.graphs[batch]=row
        return row

    @torch.no_grad()
    def forward_step(self,features,maps,hidden,*,objects):
        if self.settings!=self._settings() or self.signature!=self._signature():
            raise ValueError('Captured model storage, mode or declared risk settings changed;create a new runner')
        n=len(features);H=self.model.gru_hidden_size
        required=((features,(n,25)),(maps,(n,16,92,96)),(objects,(n,96,21)),(hidden,(1,n,H)))
        if (not 1<=n<=self.maximum_batch or any(tuple(x.shape)!=s or x.device!=self.device
                or x.dtype!=torch.float32 for x,s in required)):
            raise ValueError('Matched nonempty CUDA FP32 modalities and recurrent state required')
        size=next(b for b in self.buckets if b>=n)
        row=self.graphs.get(size) or self._capture(size)
        # Padding is zeroed so an unused old row with a bad value cannot trip
        # the finite guard. Every network operation is per world/within world.
        for name,source in (('features',features),('maps',maps),('objects',objects)):
            row[name][:n].copy_(source)
            if size>n:row[name][n:].zero_()
        row['hidden'][:,:n].copy_(hidden)
        if size>n:row['hidden'][:,n:].zero_()
        row['graph'].replay()
        if not bool(row['finite'].item()):raise FloatingPointError('CUDA graph policy produced nonfinite output')
        self.calls+=1
        return row['logits'][:n].clone(),row['values'][:n].clone(),row['following'][:,:n].clone()

    def specification(self):
        return dict(schema=SCHEMA,training_enabled=False,collector_registered=False,
            maximum_batch=self.maximum_batch,captured_buckets=sorted(self.graphs),calls=self.calls,
            optimizer_updates=0,parameters_added=0,action_override=False,
            recurrent_state_external=True,batch_padding_has_no_cross_world_operators=True,
            weights_read_from_original_in_place_storage=True,finite_output_checked_every_replay=True,
            caveat='Requires CUDA parity qualification and separate native rollout qualification before activation')
