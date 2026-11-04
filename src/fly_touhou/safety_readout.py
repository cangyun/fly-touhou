"""Training-only multilabel admissibility learning from real DN features."""
import numpy as np
import torch
from torch.nn import functional as F


def movement_trigger_weights(allowed,confidence):
    allowed=np.asarray(allowed,dtype=bool);confidence=np.asarray(confidence,dtype=np.float64)
    if (allowed.ndim!=2 or allowed.shape[1]!=18 or confidence.shape!=(len(allowed),)
            or not allowed.any(-1).all() or not np.isfinite(confidence).all() or (confidence<0).any()):
        raise ValueError('Nonempty18-way training sets and finite confidence required')
    active=confidence>0;critical=active&~allowed[:,0];ordinary=active&allowed[:,0]
    groups=[(g,share) for g,share in ((critical,.8),(ordinary,.2)) if g.any()]
    result=np.zeros(len(allowed),dtype=np.float64)
    total=sum(share for _,share in groups)
    for group,share in groups:
        result[group]=len(allowed)*share/total*confidence[group]/confidence[group].sum()
    return result


def multilabel_readout_loss(logits,allowed,teacher,weights):
    """Balance admissible/inadmissible classes inside each measured state.

    These are geometric training forecasts, not executed counterfactuals.
    No admissible mask or teacher enters the deployed actor input.
    """
    z=logits.double();positive=allowed.double();negative=(~allowed).double()
    p=-(F.logsigmoid(z)*positive).sum(-1)/positive.sum(-1).clamp_min(1.)
    n=-(F.logsigmoid(-z)*negative).sum(-1)/negative.sum(-1).clamp_min(1.)
    groups=1.+(negative.sum(-1)>0).double()
    preference=-(teacher*F.log_softmax(z,-1)).sum(-1)
    return (((p+n)/groups+.05*preference)*weights).mean()
