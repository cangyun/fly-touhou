"""Movement-only, all-candidate teacher on the student's own observations."""
import numpy as np


def label_current(encoding, snapshot, episode, *, selective=False):
    detail = encoding['details']
    m = np.asarray(detail['margins'], np.float64)
    trust = np.asarray(detail['trusted'], bool)
    safe = m[0] > 1
    usable = bool(detail['normal'] and trust[0] and safe.any())
    scores = np.zeros(18)
    for h, weight in ((0, .25), (1, .5), (2, 1.)):
        if trust[h]: scores += weight * np.tanh((m[h]-1)/8)
    route = encoding.get('route_context')
    if route is not None and trust[1]:
        if selective:scores += .5*route[:,5] + .35*route[:,6] + .25*route[:,7]
        else:scores += .15*route[:,5] + .1*route[:,7]
    # No hard lower-half preference, Boss vertical preference or item chase.
    # Unknown geometry cannot create a trusted-safe target.
    if usable: scores[~safe] = -np.inf
    meaningful=bool(usable and ((~safe).any() or np.ptp(scores[safe])>.1))
    temperature=.35 if selective else 1.
    probability = np.exp((scores-scores.max())/temperature); probability /= probability.sum()
    if selective and usable:
        probability=.9*probability+.1*safe/safe.sum()
    p = np.zeros(19, np.float32); p[:18] = probability
    return dict(schema='nbnm-current-student-selective-route-v2' if selective else 'nbnm-current-student-all-directions-v1',probabilities=p.tolist(),
        action=int(p.argmax()), guidance_weight=1. if (meaningful if selective else usable) else 0.,
        guidance_weight_reason='informative_direction_preference' if meaningful else 'no_reliable_direction_preference',
        native_episode=int(episode), frame=int(snapshot.frame),stage=int(snapshot.stage),
        stage_frame=int(snapshot.stage_frame), timing='current-before-action',
        teacher_actions=0, Bomb_target=0., source='conditional observed-geometry teacher; native outcomes override contradictions')
