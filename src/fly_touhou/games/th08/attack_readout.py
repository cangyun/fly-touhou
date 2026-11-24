"""Learned movement readout conditioned on the attack observed in original TH08."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

TASK_CONTEXT = "th08-observed-attack-v1"
PRACTICE_INTRO_CONTEXT = "th08-observed-attack-practice-v1"
OPENING_NONSPELL_CONTEXT = "th08-observed-attack-practice-opening-nonspells-v1"
PRACTICE_TASK_CONTEXTS = (PRACTICE_INTRO_CONTEXT, OPENING_NONSPELL_CONTEXT)
ATTACK_TASK_CONTEXTS = (TASK_CONTEXT, *PRACTICE_TASK_CONTEXTS)
CRITIC_TASK_CONTEXTS = ("critic-only-v1", *ATTACK_TASK_CONTEXTS)
ATTACK_AGE_COLUMN = 276
OBSERVATION_COLUMN = 280
CONTEXT_COUNT = 512
# Use the row after the active spells and two unused native result IDs.
OPENING_NONSPELL_ROWS = {4: (510, range(77, 100)), 5: (256, range(100, 120)), 6: (511, range(120, 150))}


def observed_attack_context(spell_id, result_id, boss_present, *, practice_spell=None,
                            opening_nonspell_stage=None):
    """Encode the attack from native spell history and the observed boss."""
    if spell_id is not None and 0 <= spell_id < 255:
        index = spell_id + 1
    elif practice_spell is not None and 0 <= practice_spell < 255:
        index = practice_spell + 1
    elif boss_present:
        opening = OPENING_NONSPELL_ROWS.get(opening_nonspell_stage)
        # The original can carry the preceding stage's result into the approach.
        if opening is not None and result_id not in opening[1]:
            index = opening[0]
        else:
            index = 256 + result_id if result_id is not None and 0 < result_id < 256 else 0
    else:
        index = 0
    return index / CONTEXT_COUNT


class TH08SpecializedReadout(nn.Module):
    """A trainable copy of an observed attack's existing movement readout."""

    def __init__(self, projection, context_bias, context_weight, minimum_age):
        super().__init__()
        self.projection = deepcopy(projection)
        self.context_bias = nn.Parameter(context_bias.detach().clone())
        self.context_weight = nn.Parameter(context_weight.detach().clone())
        self.scale = projection.out_features ** -.5
        self.register_buffer("minimum_age", projection.weight.new_tensor(minimum_age))

    def forward(self, activity):
        hidden = F.silu(self.projection(activity) + self.context_bias)
        return self.scale * (hidden * self.context_weight).sum(-1)


class TH08AttackReadout(nn.Module):
    """Use the same descending activity with an additional observed attack input."""

    def __init__(self, route_projection):
        super().__init__()
        width = route_projection.out_features
        device = route_projection.weight.device
        self.projection = nn.Linear(route_projection.in_features, width, device=device)
        self.projection.load_state_dict(route_projection.state_dict())
        self.context_bias = nn.Embedding(CONTEXT_COUNT, width, padding_idx=0, device=device)
        self.context_weight = nn.Embedding(CONTEXT_COUNT, width, padding_idx=0, device=device)
        nn.init.zeros_(self.context_bias.weight)
        nn.init.zeros_(self.context_weight.weight)
        self.scale = width ** -.5
        self.specialists = nn.ModuleDict()

    def add_specialist(self, context, minimum_age):
        key = str(context)
        if not 0 < context < CONTEXT_COUNT or not 0 <= minimum_age < 1 or key in self.specialists:
            raise ValueError("Choose one observed TH08 context and its normalized activation age")
        self.specialists[key] = TH08SpecializedReadout(
            self.projection, self.context_bias.weight[context], self.context_weight.weight[context], minimum_age)
        return self.specialists[key]

    def forward(self, activity, observed_context, attack_age=None):
        index = (observed_context * CONTEXT_COUNT).round().long().clamp(0, CONTEXT_COUNT - 1)
        hidden = F.silu(self.projection(activity) + self.context_bias(index)[:, None])
        scores = self.scale * (hidden * self.context_weight(index)[:, None]).sum(-1)
        if self.specialists and attack_age is None:
            raise ValueError("A specialized movement readout requires the observed attack age")
        for key, specialist in self.specialists.items():
            enabled = (index == int(key)) & (attack_age >= specialist.minimum_age)
            scores = torch.where(enabled[:, None], specialist(activity), scores)
        return scores
