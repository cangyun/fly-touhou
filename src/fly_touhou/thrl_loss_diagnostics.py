"""Detached output-gradient diagnostics; no optimizer, sampling or updates."""


def teacher_fit_statistics(logits, targets, weights=None):
    """Separate target entropy from imitation KL on the actual legal actions."""
    import torch
    from .thrl_policy_actions import policy_logits
    with torch.no_grad():
        clean=policy_logits(logits.detach())
        q=torch.as_tensor(targets,device=clean.device,dtype=clean.dtype).detach()
        if q.shape!=clean.shape or not torch.isfinite(q).all() or (q<0).any():
            raise ValueError('Finite paired teacher probabilities required')
        legal=torch.isfinite(clean);q=q*legal.to(q.dtype)
        mass=q.sum(-1,keepdim=True)
        if (mass<=0).any():raise ValueError('Teacher has no legal probability mass')
        q=q/mass;positive=q>0
        logq=torch.where(positive,q.clamp_min(torch.finfo(q.dtype).tiny).log(),0.)
        logp=clean.log_softmax(-1).masked_fill(~legal,0.)
        entropy=-(q*logq).sum(-1)
        ce=-(q*logp).sum(-1)
        w=torch.ones_like(entropy) if weights is None else torch.as_tensor(weights,device=clean.device,dtype=clean.dtype).detach()
        if w.shape!=entropy.shape or not torch.isfinite(w).all() or not ((w==0)|(w==1)).all():
            raise ValueError('Paired0/1 teacher confidence required')
        n=int(w.sum())
        mean=lambda x:float((x*w).sum()/n) if n else None
        return dict(states=n,target_entropy=mean(entropy),cross_entropy=mean(ce),
                    teacher_to_policy_KL=mean(ce-entropy),legal_actions_conditioned=True)


def output_gradient_statistics(logits, losses):
    """Weighted loss effects at actor outputs, not shared parameter gradients.

    autograd.grad leaves parameter .grad fields, graph values and RNG intact;
    retain_graph lets the one ordinary training backward proceed unchanged.
    """
    import torch
    if logits.shape[-1]!=19 or not logits.requires_grad:
        raise ValueError('Differentiable actual19-action policy outputs required')
    vectors={};detail={}
    for name,loss in losses.items():
        if loss.numel()!=1 or not torch.isfinite(loss.detach()):
            raise ValueError('A finite scalar weighted loss is required')
        gradient=(torch.autograd.grad(loss,logits,retain_graph=True,allow_unused=True)[0]
                  if loss.requires_grad else None)
        vector=torch.zeros_like(logits) if gradient is None else gradient.detach()
        if not torch.isfinite(vector).all():raise FloatingPointError('Nonfinite diagnostic actor gradient')
        vectors[name]=vector.reshape(-1,19)
        g=vectors[name]
        detail[name]=dict(L2=float(g.norm()),movement_L2=float(g[:,:18].norm()),Bomb_L2=float(g[:,18].norm()))
    ppo=vectors.get('PPO_actor')
    if ppo is not None:
        norm=ppo.norm()
        for name,gradient in vectors.items():
            if name=='PPO_actor':continue
            denominator=norm*gradient.norm()
            detail[name]['cosine_with_PPO_actor']=(float((ppo*gradient).sum()/denominator) if denominator>0 else None)
            detail[name]['L2_relative_to_PPO_actor']=float(gradient.norm()/norm) if norm>0 else None
    return dict(schema='B1-detached-weighted-actor-output-gradients-v1',components=detail,
        scope='Actual minibatch actor-output derivatives;not full shared-parameter gradients or native causal effects',
        optimizer_updates=0,sampling_calls=0,parameter_grad_fields_modified=False)
