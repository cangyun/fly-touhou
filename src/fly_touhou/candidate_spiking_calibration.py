"""Calibrate a learned candidate SNN's DN-only linear risk readout.

The frozen replay is a diagnostic optimization phase, not a new brain model.
All labels come from TRAIN worlds; held-out worlds only select ridge strength.
Neither fitted nor deployed heads receive geometric cues directly.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .candidate_spiking_model import candidates
from .compact_spiking_data import Streams
from .spiking_engineered_control import SpikingState


@torch.no_grad()
def collect_dn_queries(model, records, *, seed, progress=lambda x: None):
    stream = Streams(records, seed=seed)
    state = model.initial_state(stream.width)
    gathered = {k: [] for k in ('DN', 'margins', 'mask', 'before_scores')}
    chunks = 0
    model.eval()
    rng = torch.Generator(device=model.device).manual_seed(seed)
    while True:
        raw = stream.next()
        data = {k: torch.from_numpy(v).to(model.device) for k, v in raw.items()}
        for t in range(stream.length):
            state = state.reset(data['episode_starts'][t])
            cue = candidates(data['features'][t])
            b = cue.shape[0]
            index = torch.arange(b, device=model.device)
            actual = data['actions'][t]
            if bool(((actual < 0) | (actual > 17)).any()):
                raise ValueError('Only recorded NoBomb source actions can carry replay history')
            uniform = torch.ones((b, 18), device=model.device)
            danger = (cue[..., 0] > .124).float()
            boundary = (cue[..., 0].abs() < .8).float()
            pools = [uniform, torch.where(danger.sum(-1, keepdim=True) > 0, danger, uniform),
                     torch.where(boundary.sum(-1, keepdim=True) > 0, boundary, uniform)]
            queries = [torch.multinomial(p, 1, generator=rng)[:, 0] for p in pools]
            encoded = torch.cat([cue[index, q] for q in [*queries, actual]])
            before = SpikingState(*(v.repeat(4, 1) for v in state.tensors()))
            count = len(encoded)
            # Match the deployed sparse numerical batch; unused lanes are discarded.
            if count < 128:
                encoded = torch.cat((encoded, encoded.new_zeros(128-count, 8)))
                zero = model.initial_state(128-count)
                before = SpikingState(*(torch.cat((v, z)) for v, z in zip(before.tensors(), zero.tensors())))
            DN, following = model.core(encoded, before)
            state = SpikingState(*(v[3*b:4*b] for v in following.tensors()))
            valid = data['valid'][t] & data['unprotected_mask'][t]
            for j, q in enumerate(queries):
                selected = DN[j*b:(j+1)*b]
                margins = data['margins'][t].gather(-1, q[:, None, None].expand(b, 3, 1))[..., 0]
                mask = valid[:, None] & data['trusted'][t]
                for key, value in dict(DN=selected, margins=margins, mask=mask,
                                       before_scores=model.risk(selected)).items():
                    gathered[key].append(value.cpu())
        chunks += 1
        progress(dict(chunks=chunks, recorded_rows=chunks*stream.width*stream.length))
        if not stream.queue and all(slot is None for slot in stream.slots):
            break
    return {k: torch.cat(v) for k, v in gathered.items()}


def pool_weights(margins, mask):
    """Global class pools prevent rare danger queries disappearing in safe scenes."""
    result = torch.zeros_like(margins)
    for fraction, pool in ((.25, mask & (margins <= 0)),
                           (.5, mask & (margins.abs() < 8)), (.25, mask)):
        pool = pool if bool(pool.any()) else mask
        result += fraction * pool / pool.sum().clamp_min(1)
    return result


def metrics(scores, margins, mask):
    total = scores.sum()*0
    for horizon, weight in enumerate((1., .3, .1)):
        m = margins[:, horizon]
        s = scores[:, horizon]
        valid = mask[:, horizon]
        known = (m <= 0) | (m > 1)
        regression = F.smooth_l1_loss(s.tanh(), torch.tanh((1-m)/8), reduction='none', beta=.1)
        classification = F.binary_cross_entropy_with_logits(8*s-1, (m <= 0).float(), reduction='none')
        total += weight * ((pool_weights(m, valid)*regression).sum()
                           + .2*(pool_weights(m, valid & known)*classification).sum())
    danger = mask[:, 0] & (margins[:, 0] <= 0)
    safe = mask[:, 0] & (margins[:, 0] > 1)
    predicted = scores[:, 0] >= .125
    return dict(balanced_loss=float(total.detach()), danger_queries=int(danger.sum()),
                danger_recall=float((predicted & danger).sum()/danger.sum().clamp_min(1)),
                safe_queries=int(safe.sum()),
                safe_false_alarm=float((predicted & safe).sum()/safe.sum().clamp_min(1)))


@torch.no_grad()
def fit_readout(train, held, *, device, strengths=(.0001, .001, .01, .1), progress=lambda x: None):
    """Weighted ridge; transforms the solution back to a raw-DN affine head."""
    x = train['DN'].to(device, torch.float64)
    y = train['margins'].to(device, torch.float64)
    mask = train['mask'].to(device)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    z = torch.cat(((x-mean)/std, x.new_ones((len(x), 1))), -1)
    gram = []; rhs = []
    for h in range(3):
        weights = pool_weights(y[:, h], mask[:, h])
        # Saturated far-away labels must not dominate the near-collision fit.
        target = ((1-y[:, h])/8).clamp(-2., 1.5)
        gram.append(z.T @ (weights[:, None]*z))
        rhs.append(z.T @ (weights*target))
    hx = held['DN'].to(device)
    hm = held['margins'].to(device)
    hmask = held['mask'].to(device)
    candidates_ = []
    for strength in strengths:
        coefficients = []
        for matrix, b in zip(gram, rhs):
            a = matrix.clone()
            a.diagonal()[:-1].add_(strength)
            # The intercept remains effectively unregularized.
            a[-1, -1] += 1e-10
            coefficients.append(torch.linalg.solve(a, b))
        c = torch.stack(coefficients)
        weight = (c[:, :-1]/std).float()
        bias = (c[:, -1]-(c[:, :-1]*mean/std).sum(-1)).float()
        report = metrics(F.linear(hx, weight, bias), hm, hmask)
        candidates_.append(dict(strength=strength, weight=weight.cpu(), bias=bias.cpu(), heldout=report))
        progress(dict(strength=strength, **report))
    selected = min(candidates_, key=lambda c: c['heldout']['balanced_loss'])
    return selected, [{k: v for k, v in c.items() if k not in ('weight', 'bias')} for c in candidates_]
