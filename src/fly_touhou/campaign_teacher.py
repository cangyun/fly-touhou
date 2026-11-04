"""Inspectable curriculum teacher; never a substitute for the full-brain actor."""
from __future__ import annotations
import numpy as np
from .planner import CollisionPlanner
from .state import Snapshot


class CampaignTeacher:
    def __init__(self, planner: CollisionPlanner):
        self.planner = planner

    def probabilities(self, snapshot: Snapshot) -> np.ndarray:
        movement = self.planner.action_probabilities(snapshot)
        can_bomb = snapshot.bombs > 0 and snapshot.player is not None
        bomb_probability = 0.0
        if can_bomb and snapshot.player.invulnerable_frames <= 0:
            best = int(np.argmax(movement))
            clearance = self.planner.action_metrics(snapshot, best).clearance
            if snapshot.player.state == 4 or clearance < 0:
                bomb_probability = 0.995
        probabilities = np.concatenate(((1-bomb_probability)*movement,
                                         bomb_probability*movement)).astype(np.float32)
        return probabilities / probabilities.sum()

    def reset(self):
        self.planner.reset()
