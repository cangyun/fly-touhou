"""Cycle 073 frozen readouts for the recorded 407-column observations.

The shared 138,639-neuron LIF core, 18 candidate updates, three value heads and
state commitment use the existing implementation from this tagged checkout.
"""
import torch
from torch import nn
from torch.nn import functional as F

from fly_touhou.candidate_spiking_model import build_compact_model
from fly_touhou.compact_story_model import CompactStoryPolicy, ContextPortDrive
from fly_touhou.dn_residual_risk import load_head
from fly_touhou.nbnm_model import configure_model

from safe_idle import apply_idle_bonus


class CorridorPortDrive(ContextPortDrive):
    def _drive(self, x):
        drive = self.base(x[:, :8]) + self.context(x[:, 8:47])
        drive = drive + self.extended_context(x[:, 47:63])
        drive = drive + self.corridor_context(x[:, 63:70])
        return F.relu(drive)


class FrozenCorridorPolicy(CompactStoryPolicy):
    def candidate_features(self, features):
        # The first 281 columns use the existing task mask for the actor.
        base = super().candidate_features(features[..., :281])
        corridor = features[..., 281:].reshape(*features.shape[:-1], 18, 7)
        return torch.cat((base, corridor), dim=-1)

    def heads(self, dn, features):
        out = super().heads(dn, features)
        candidates = dn.reshape(len(features), 18, -1)
        normalized = F.layer_norm(candidates, (candidates.shape[-1],))
        risk = self.corridor_head(normalized).sigmoid()
        witness = self.corridor_witness_head(normalized).sigmoid()
        vulnerable = (features[:, 14] > 0.5) & (features[:, 5] <= 0) & (features[:, 13] <= 0)
        observed = features[..., 281:].reshape(-1, 18, 7)[..., 6].amax(-1) > 0
        active = (vulnerable & observed)[..., None]
        risk_cost = (0.65 * risk[..., 0] + 0.35 * risk[..., 1]) * active
        weight = self.nbnm["corridor_risk"]
        movement = out["logits"][..., :18] - weight["actor_weight"] * (risk_cost - risk_cost.mean(-1, keepdim=True))
        exit_support = 0.4 * witness[..., 0] + 0.6 * witness[..., 1]
        delta = (exit_support - exit_support.mean(-1, keepdim=True)) * active
        movement = movement + weight["witness_weight"] * delta
        logits = torch.cat((movement, out["logits"][..., 18:]), dim=-1)
        out["logits"] = apply_idle_bonus(logits, features, self.safe_idle_bonus) if self.safe_idle_bonus else logits
        return out


def load_frozen(connectome, checkpoint, bonus):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["input_schema"] != "connectome-nbnm-route407-corridor-v1" or not saved["bombs_disabled"]:
        raise ValueError("Expected the released 407-column NBNM checkpoint")
    base = build_compact_model(connectome, seed=saved["seed"], device="cuda:0")
    risk = {k.removeprefix("risk."): v for k, v in saved["state_dict"].items() if k.startswith("risk.")}
    base.risk = load_head(dict(schema=saved["risk_readout_kind"], state_dict=risk, width=saved["risk_readout_width"],
                               input_soft_clip=saved["risk_input_soft_clip"], base_input_soft_clip=saved["risk_base_input_soft_clip"],
                               baseline_trainable=saved["trainable_risk_baseline"]), device=base.device)
    model = FrozenCorridorPolicy(base, temperature=saved["temperature"])
    model.core.input_encoder = CorridorPortDrive(model.core.input_encoder.base).to(model.device)
    configure_model(model, saved["nbnm"], reset_critics=False)
    encoder = model.core.input_encoder
    encoder.corridor_context = nn.Linear(7, encoder.residual.out_features, bias=False, device=model.device)
    width = 2 * len(model.core.output_indices)
    model.corridor_head = nn.Linear(width, 2, device=model.device)
    model.corridor_witness_head = nn.Linear(width, 2, device=model.device)
    model.core.sensory_width = 70
    model.input_schema = saved["input_schema"]
    model.load_state_dict(saved["state_dict"], strict=True)
    for name in ("phase_valid_control", "protection_aware_risk", "protection_expiry_inclusive", "risk_policy_gradient",
                 "neural_risk_scale", "idle_prior_scale", "soft_risk_composition", "pad_candidate_batch"):
        setattr(model, name, saved[name])
    model.core.input_encoder.batch_invariant = saved["batch_invariant_sensory"]
    model.core.sparse.deterministic_spmm = saved["deterministic_spmm"]
    model.safe_idle_bonus = bonus
    return model.eval()
