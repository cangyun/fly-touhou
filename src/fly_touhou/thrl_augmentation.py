"""Horizontal symmetry of separated native observations and source19 actions.

Training augmentation only. No native counterfactual outcome is fabricated and
PPO rollout probabilities/actions must never be changed with this transform.
"""
from __future__ import annotations


def mirror_action_indices():
    from .state import decode_action
    from .thrl_native_env import SOURCE19_TO_NATIVE36
    directions=[decode_action(a) for a in SOURCE19_TO_NATIVE36[:18]]
    lookup={tuple(x):i for i,x in enumerate(directions)}
    return [lookup[(-x,y,focus)] for x,y,focus in directions]+[18]


def horizontal_mirror(features,maps,objects,targets,selected):
    """Mirror whole sequences[B,L,...], keeping source tensors immutable."""
    import torch
    if features.ndim!=3 or features.shape[-1]!=25 or maps.shape[:2]!=features.shape[:2]:
        raise ValueError('Horizontal augmentation requires separated25 sequences')
    if (objects.shape[:-1]!=(*features.shape[:2],96) or objects.shape[-1] not in (21,29)
            or targets.shape!=(*features.shape[:2],19)):
        raise ValueError('Object/target sequences must match separated observations')
    selected=torch.as_tensor(selected,device=features.device,dtype=torch.bool)
    if selected.shape!=(features.shape[0],):raise ValueError('Mirror selection must apply to whole sequences')
    if not selected.any():return features,maps,objects,targets
    f,m,o,y=(v.clone() for v in (features,maps,objects,targets))
    f[selected,:,0]=1-f[selected,:,0]
    f[selected,:,16]=-f[selected,:,16]  # measured asymmetric player x offset
    m[selected]=maps[selected].flip(-1)
    for channel in (1,6,11):m[selected,:,channel]=-m[selected,:,channel]
    o[selected,:,:,6]=-o[selected,:,:,6]
    o[selected,:,:,8]=-o[selected,:,:,8]
    if objects.shape[-1]==29:
        o[selected,:,:,21]=-o[selected,:,:,21]
        o[selected,:,:,23]=-o[selected,:,:,23]
    permutation=torch.tensor(mirror_action_indices(),device=targets.device)
    y[selected]=targets[selected][...,permutation]
    return f,m,o,y
