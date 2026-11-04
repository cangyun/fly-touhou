"""The distance-triggered marker/EX restart in TH14 callback42cb00.

This models the callback only. It does not yet model the restarted bullet
program or certify a future player path. No production sensor uses it yet.
424866..424872 passes EnemyData as ECX; its+29c/+2a0 are the two radii.
"""
import numpy as np

CALLBACK = 0x42cb00


def marker_transition(marker, timer, bullet_position, player_position, near_radius, far_radius):
    f = np.float32
    delta = np.asarray(bullet_position, dtype=np.float32)-np.asarray(player_position, dtype=np.float32)
    distance2 = f(f(delta[0]*delta[0])+f(delta[1]*delta[1]))
    restart = None
    if marker == 1 and distance2 < f(f(near_radius)*f(near_radius)):
        marker, timer, restart = 2, 0, 5
    elif marker == 2:
        timer = (int(timer)+1+2**31) % 2**32-2**31
        if distance2 >= f(f(far_radius)*f(far_radius)):
            marker, restart = 1, 14
    return dict(marker=int(marker), timer=int(timer), restart_ex_index=restart,
                clear_ex_active_flags=restart is not None)
