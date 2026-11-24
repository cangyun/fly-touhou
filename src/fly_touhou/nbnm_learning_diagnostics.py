"""Read-only channel scales for the NBNM learner; never change its targets."""
import torch


@torch.no_grad()
def target_summary(data, advantages, returns, objective_weights):
    device = advantages.device
    valid = data.get('valid_mask')
    valid = (torch.ones(advantages.shape[:-1], dtype=torch.bool, device=device)
             if valid is None else valid.to(device))
    a = advantages[valid]
    r = returns[valid]
    rewards = data['reward_vectors'].to(device)[valid]
    old = data['old_values'].to(device)[valid]
    weights = data.get('objective_weights')
    weights = (a.new_tensor(objective_weights).expand_as(a) if weights is None
               else weights.to(device)[valid])
    weighted = a * weights
    channels = {}
    for i, name in enumerate(('completion', 'speed', 'progress')):
        channels[name] = dict(
            reward_sum=float(rewards[:, i].sum()),
            reward_rows=int((rewards[:, i] != 0).sum()),
            old_value_mean=float(old[:, i].mean()),
            return_mean=float(r[:, i].mean()),
            return_std=float(r[:, i].std(unbiased=False)),
            advantage_std=float(a[:, i].std(unbiased=False)),
            weighted_advantage_std=float(weighted[:, i].std(unbiased=False)),
            actor_weight_min=float(weights[:, i].min()),
            actor_weight_max=float(weights[:, i].max()))
    return dict(real_rows=len(a), channels=channels,
                combined_advantage_std=float(weighted.sum(-1).std(unbiased=False)))
