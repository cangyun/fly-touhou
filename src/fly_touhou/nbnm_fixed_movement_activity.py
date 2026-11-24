"""Reuse fixed descending activity when fitting movement readouts."""

import torch


@torch.no_grad()
def capture_movement_activity(model, features, starts, actions, state):
    """Replay the recorded actions and save candidate DN inputs to the heads."""
    readouts = ("route.", "corridor_head.", "corridor_witness_head.")
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith(readouts)):
        raise ValueError("Only movement and corridor readouts may be trainable while reusing descending activity")
    rows = []
    for frame, start, action in zip(features, starts, actions, strict=True):
        output, state, shared = model.step(
            frame, state.reset(start), actions=action, return_all_DN=True,
        )
        if model.shared_directions is None:
            activity = output["candidate_DN"]
            pooled = None
        else:
            activity = model.shared_directions.directions(shared)
            pooled = shared
        rows.append((activity.detach(), None if pooled is None else pooled.detach()))
    return rows


def movement_logits_from_activity(model, features, actions, activity):
    """Use the same heads and explicit action tape as the recurrent training path."""
    logits = []
    for frame, action, (descending, pooled) in zip(features, actions, activity, strict=True):
        output = model.heads(descending.to(frame.device), frame,
                             pooled_DN=None if pooled is None else pooled.to(frame.device))
        model._choose(output, frame, actions=action, sample=False,
                      action_selector=None, previous_actions=None)
        logits.append(output["logits"])
    return torch.stack(logits)
