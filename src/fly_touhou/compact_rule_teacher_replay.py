"""Movement-only imitation from partial, full-game rule-teacher archives."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def mask_rule_guidance_on_contact(weights, actions, outcomes, unprotected, elapsed_frames=None):
    """Do not imitate a teacher on an action that actually made contact.

    Only the executed movement is an observed outcome. Other movements are
    not relabelled as safe, and the original collision target is retained.
    """
    contact=unprotected & (actions>=0) & (actions<18) & (outcomes==1)
    if elapsed_frames is not None:
        contact=contact & (elapsed_frames>0) & (elapsed_frames<=4)
    return torch.where(contact,torch.zeros_like(weights),weights), contact


def movement_distribution_loss(logits, probabilities, weights, actions, controllable):
    """Match the teacher's entire conditional 18-movement distribution.

    Recorded Bomb decisions are deliberately excluded.  A policy preference is
    not a verified per-direction collision risk, so this is not a risk label.
    """
    movement = logits[..., :18]
    teacher = probabilities[..., :18].clamp_min(0.)
    legal = torch.isfinite(movement)
    teacher = teacher * legal.to(teacher.dtype)
    mass = teacher.sum(-1, keepdim=True)
    teacher = teacher / mass.clamp_min(1e-12)
    logp = movement.masked_fill(~legal, -1e9).log_softmax(-1)
    row_loss = -(teacher * logp).sum(-1)
    eligible = (weights > 0) & (actions < 18) & controllable & (mass[..., 0] > 0)
    w = torch.where(eligible, weights, 0.)
    return (row_loss * w).sum() / w.sum().clamp_min(1.), eligible


class RuleTeacherArchiveReplay:
    def __init__(self, manifest, *, split='train'):
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'partial-v5-resource-v10-teacher-stage-window-v1':
            raise ValueError('Expected converted ResourceV10 rule-teacher windows')
        self.burn_in = int(self.manifest['burn_in'])
        if split not in ('train', 'heldout'):
            raise ValueError('Teacher windows must be train or heldout')
        self.split = split
        candidates = [w for w in self.manifest['windows'] if w['split'] == split]
        self.windows = []
        for row in candidates:
            with np.load(self.manifest_path.parent / row['file'], allow_pickle=False) as data:
                f = data['features'][self.burn_in:]
                eligible = ((data['teacher_guidance_weight'][self.burn_in:] > 0)
                    & (data['actions'][self.burn_in:] < 18)
                    & ((f[:, 14] > .5) | (f[:, 17] > .5))
                    & (data['targets'][self.burn_in:, :18].sum(-1) > 0))
            if eligible.any():
                self.windows.append(row)
        self.empty_windows_excluded = len(candidates)-len(self.windows)
        if not self.windows:
            raise ValueError('The teacher archive contains no training windows')

    def loss(self, model, *, index):
        from .compact_story_model import sequence
        row = self.windows[index % len(self.windows)]
        with np.load(self.manifest_path.parent / row['file'], allow_pickle=False) as archive:
            data = {k: torch.as_tensor(archive[k].copy(), device=model.device)[:, None]
                    for k in ('features', 'targets', 'actions', 'episode_starts',
                              'teacher_guidance_weight')}
        # The historical window has no recorded fly state.  Warm it with the
        # actual teacher actions; it is approximate, not a forged history.
        state = model.initial_state(1)
        if self.burn_in:
            with torch.no_grad():
                state = sequence(model, data['features'][:self.burn_in],
                    data['episode_starts'][:self.burn_in], state,
                    data['actions'][:self.burn_in], checkpointed=False)['state'].detach()
        result = sequence(model, data['features'][self.burn_in:],
            data['episode_starts'][self.burn_in:], state, data['actions'][self.burn_in:])
        f = data['features'][self.burn_in:]
        controllable = (f[..., 14] > .5) | (f[..., 17] > .5)
        loss, eligible = movement_distribution_loss(result['logits'],
            data['targets'][self.burn_in:], data['teacher_guidance_weight'][self.burn_in:],
            data['actions'][self.burn_in:], controllable)
        return loss, dict(rule_archive_stage=row['stage'], rule_archive_world=row['world'],
            rule_archive_teacher_rows=int(eligible.sum()), rule_archive_PPO_rows=0,
            rule_archive_burn_in=self.burn_in, rule_archive_movement_loss=float(loss.detach()))

    def summary(self):
        return dict(manifest=str(self.manifest_path), split=self.split, windows=len(self.windows),
            empty_windows_excluded=self.empty_windows_excluded,
            stage3=sum(w['stage'] == 3 for w in self.windows),
            stage4=sum(w['stage'] == 4 for w in self.windows),
            heldout_training=False, teacher='ResourceV10', teacher_plays_training_games=False,
            archived_actions_are_PPO_samples=False, approximate_initial_brain_state=True,
            supervision='conditional 18-way movement preferences; archived Bomb rows excluded',
            native_risk_labels=False, sensor_limitations=self.manifest['sensor_limitations'])
