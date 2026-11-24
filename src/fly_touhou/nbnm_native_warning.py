"""Estimate 48-frame contact on the student's own continuation after a first move."""
import math

import torch
from torch import nn


REVISION = 'native-policy-warning48-v1'
WIDTH = 10
DN_REVISION = 'native-DN-policy-warning48-v1'


def predictors(probability, route):
    """Use only descending-neuron readouts available before choosing the action."""
    shape = probability.shape
    best = probability.amin(dim=-2, keepdim=True).expand(shape)
    mean = probability.mean(dim=-2, keepdim=True).expand(shape)
    relative_route = (route - route.mean(dim=-1, keepdim=True)).unsqueeze(-1)
    return torch.cat((probability, best, mean, relative_route), dim=-1)


def validate(settings):
    if settings is None:
        return
    weights = settings.get('initial_weights', ())
    bias = settings.get('initial_bias')
    revision = settings.get('revision')
    if (revision not in (REVISION, DN_REVISION) or len(weights) < WIDTH
            or (revision == REVISION and len(weights) != WIDTH)
            or (revision == DN_REVISION and len(weights) == WIDTH)
            or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in [*weights, bias])
            or not 0 <= settings.get('actor_weight', 0) <= 1
            or not 0 < settings.get('loss_weight', .1) <= 1):
        raise ValueError('Invalid fitted native 48-frame warning configuration')


class NativeWarning48(nn.Linear):
    def __init__(self, settings, device, dn_width):
        validate(settings)
        self.read_DN = settings['revision'] == DN_REVISION
        width = WIDTH + (dn_width if self.read_DN else 0)
        if len(settings['initial_weights']) != width:
            raise ValueError('Native warning must match the actual descending-neuron readout width')
        super().__init__(width, 1, device=device)
        with torch.no_grad():
            self.weight.copy_(self.weight.new_tensor(settings['initial_weights']))
            self.bias.fill_(settings['initial_bias'])

    def forward(self, probability, route, descending):
        # A rare contact cannot rewrite the three immediate-risk heads by this loss.
        x = predictors(probability.detach(), route.detach())
        if self.read_DN:
            x = torch.cat((x, descending.detach()), dim=-1)
        return super().forward(x).squeeze(-1)


def selected_loss(warning_logits, actions, known, hits, normalizer):
    selected = warning_logits.gather(-1, actions.clamp(0, 17).unsqueeze(-1)).squeeze(-1)
    losses = nn.functional.binary_cross_entropy_with_logits(selected, hits.to(selected.dtype), reduction='none')
    return (losses * known).sum() / normalizer
