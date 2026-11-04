"""Sample-weighted microchunk gradients at one unchanged parameter state.

This helper does not choose data, calculate losses, clip, or step an optimizer.
It lets the caller retain one recurrent chunk's activation memory at a time.
"""
import torch


class SerialGradientMean:
    def __init__(self,parameters):
        self.parameters=tuple(parameters)
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError('Nonempty trainable parameter sequence required')
        if len({id(p) for p in self.parameters})!=len(self.parameters):
            raise ValueError('A parameter cannot occur twice')
        self.versions=tuple(p._version for p in self.parameters)
        self.sums=[torch.zeros_like(p,dtype=torch.float64) for p in self.parameters]
        self.seen=[False]*len(self.parameters)
        self.samples=0
        self.chunks=0

    def _unchanged(self):
        if tuple(p._version for p in self.parameters)!=self.versions:
            raise RuntimeError('All microchunk gradients must use identical parameters')

    @torch.no_grad()
    def add(self,samples):
        self._unchanged()
        if isinstance(samples,bool) or not isinstance(samples,int) or samples<1:
            raise ValueError('A positive integer microchunk sample count is required')
        for p in self.parameters:
            if p.grad is None:continue
            if p.grad.is_sparse or p.grad.shape!=p.shape or not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError('Finite dense microchunk gradients required')
        for i,(p,total) in enumerate(zip(self.parameters,self.sums)):
            if p.grad is None:continue
            total.add_(p.grad.detach().double(),alpha=samples)
            self.seen[i]=True
        self.samples+=samples
        self.chunks+=1

    @torch.no_grad()
    def assign_mean(self):
        self._unchanged()
        if not self.samples:raise ValueError('No microchunk gradients were accumulated')
        for p,total,seen in zip(self.parameters,self.sums,self.seen):
            p.grad=(total/self.samples).to(dtype=p.dtype) if seen else None
        return {'microchunks':self.chunks,'samples':self.samples,
            'accumulation_dtype':'float64','optimizer_step_executed':False}
