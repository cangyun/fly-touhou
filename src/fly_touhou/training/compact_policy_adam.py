"""Keep sub-float32 policy updates across Adam steps and checkpoints.

The model and its gradients remain float32. Policy Adam arithmetic uses a
float64 master copy, including moments, so a small update is accumulated
instead of being discarded each time it is rounded for the next forward pass.
The three critics retain their ordinary Adam precision and learning rate.
"""
from __future__ import annotations

import torch


class PolicyMasterAdam(torch.optim.Adam):
    precision_revision = 'compact-policy-master-fp64-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy_masters = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        swapped = []
        try:
            for group in self.param_groups:
                if group.get('name') == 'three_critics':
                    continue
                for parameter in group['params']:
                    if parameter.grad is None or parameter.dtype != torch.float32:
                        continue
                    master = self._policy_masters.get(parameter)
                    if master is None:
                        master = parameter.detach().to(torch.float64)
                        self._policy_masters[parameter] = master
                    # Legacy checkpoints have float32 moments. Preserve their
                    # values and step counter when promoting the arithmetic.
                    state = self.state[parameter]
                    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                        if name in state and state[name].dtype != torch.float64:
                            state[name] = state[name].to(torch.float64)
                    data, gradient = parameter.data, parameter.grad
                    swapped.append((parameter, data, gradient))
                    parameter.grad = None
                    parameter.data = master
                    parameter.grad = gradient.to(torch.float64)
            # Use the installed public Adam implementation, including its
            # normal lazy state initialization, bias correction and eps rules.
            super().step()
        finally:
            for parameter, data, gradient in swapped:
                parameter.grad = None
                parameter.data = data
                parameter.copy_(self._policy_masters[parameter])
                parameter.grad = gradient
        return loss

    def state_dict(self):
        result = super().state_dict()
        result['policy_master_precision'] = self.precision_revision
        result['policy_masters'] = {
            saved_id: self._policy_masters[parameter]
            for saved_group, group in zip(result['param_groups'], self.param_groups)
            for saved_id, parameter in zip(saved_group['params'], group['params'])
            if parameter in self._policy_masters
        }
        return result

    def reset_parameter_state(self, parameter):
        """Use after an explicit head refit replaces this model parameter."""
        self.state.pop(parameter, None)
        self._policy_masters.pop(parameter, None)

    @torch.no_grad()
    def sampled_update_snapshot(self, per_parameter=4096):
        """Small deterministic samples for reporting; never gate an update."""
        snapshot = []
        for group in self.param_groups:
            if group.get('name') == 'three_critics':
                continue
            for index, parameter in enumerate(group['params']):
                stride = max(1, (parameter.numel() + per_parameter - 1) // per_parameter)
                visible = parameter.detach().reshape(-1)[::stride].double().clone()
                master = self._policy_masters.get(parameter, parameter.detach())
                master = master.reshape(-1)[::stride].double().clone()
                snapshot.append((group['name'], index, parameter, stride, visible, master))
        return snapshot

    @torch.no_grad()
    def sampled_update_changes(self, snapshot):
        result = {}
        for group, index, parameter, stride, before, master_before in snapshot:
            visible = parameter.detach().reshape(-1)[::stride].double()
            master = self._policy_masters.get(parameter, parameter.detach())
            master = master.reshape(-1)[::stride].double()
            values = torch.stack(((visible != before).double().mean(),
                (master != master_before).double().mean(), (visible - before).abs().mean(),
                (master - master_before).abs().mean(), (master - visible).abs().mean())).cpu().tolist()
            result[f'{group}/parameter{index}'] = dict(total_parameters=parameter.numel(),
                sampled_parameters=visible.numel(), model_changed_fraction=values[0],
                master_changed_fraction=values[1], mean_abs_model_update=values[2],
                mean_abs_master_update=values[3], mean_abs_rounding_residue=values[4])
        return result

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._policy_masters = {}
        masters = state_dict.get('policy_masters', {})
        for saved_group, group in zip(state_dict['param_groups'], self.param_groups):
            if group.get('name') == 'three_critics':
                continue
            for saved_id, parameter in zip(saved_group['params'], group['params']):
                # Adam's generic loader casts state to the model dtype. Restore
                # the saved precision directly, without an intermediate cast.
                saved_state = state_dict['state'].get(saved_id, {})
                for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if name in saved_state:
                        self.state[parameter][name] = saved_state[name].to(
                            device=parameter.device, dtype=torch.float64, copy=True)
                if saved_id in masters:
                    self._policy_masters[parameter] = masters[saved_id].to(
                        device=parameter.device, dtype=torch.float64, copy=True)
