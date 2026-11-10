"""Movement-only, all-candidate teacher on the student's own observations."""
import numpy as np


def label_current(encoding, snapshot, episode):
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
        scores += .15*route[:,5] + .1*route[:,7]
    # No hard lower-half preference, Boss vertical preference or item chase.
    # Unknown geometry cannot create a trusted-safe target.
    if usable: scores[~safe] = -np.inf
    probability = np.exp(scores-scores.max()); probability /= probability.sum()
    p = np.zeros(19, np.float32); p[:18] = probability
    return dict(schema='nbnm-current-student-all-directions-v1',probabilities=p.tolist(),
        action=int(p.argmax()), guidance_weight=1. if usable else 0.,
        native_episode=int(episode), frame=int(snapshot.frame),stage=int(snapshot.stage),
        stage_frame=int(snapshot.stage_frame), timing='current-before-action',
        teacher_actions=0, Bomb_target=0., source='conditional observed-geometry teacher; native outcomes override contradictions')
