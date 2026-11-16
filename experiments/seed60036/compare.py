"""Recalculate the idle and direction-change counts from both recorded policies.

Only NumPy is required. The two NPZ members contain independent trajectories.
"""

import argparse
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np


from safe_idle import apply_idle_bonus, settled_safe24


def load_run(archive, variant):
    if variant not in ("baseline", "idle025"):
        raise ValueError("Unknown variant")
    with ZipFile(archive) as bundle, np.load(BytesIO(bundle.read(variant + ".npz")), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def count_run(run):
    actions = run["actions"]
    if not np.array_equal(actions, run["logits"].argmax(axis=1)):
        raise ValueError("Recorded actions disagree with final movement scores")
    if not np.array_equal(actions, run["probabilities"].argmax(axis=1)):
        raise ValueError("Recorded actions disagree with final movement probabilities")
    np.testing.assert_allclose(run["probabilities"].sum(axis=1), 1, rtol=0, atol=3e-6)
    directions = actions % 9
    idle = directions == 0
    gate = settled_safe24(run["features"])
    dt = np.diff(run["frames"])
    same_stage = run["stages"][1:] == run["stages"][:-1]
    moving_pairs = same_stage & (dt > 0) & (dt <= 4) & ~idle[:-1] & ~idle[1:]
    turned = moving_pairs & (directions[:-1] != directions[1:])
    per_stage = []
    for stage in range(1, 7):
        group = run["stages"] == stage
        per_stage.append(dict(stage=stage, decisions=int(group.sum()), safe_decisions=int((gate & group).sum()),
                              safe_idle=int((idle & gate & group).sum()), moving_pairs=int((moving_pairs & group[:-1]).sum()),
                              turns=int((turned & group[:-1]).sum())))
    return dict(decisions=len(actions), safe_decisions=int(gate.sum()), safe_idle=int((gate & idle).sum()),
                moving_pairs=int(moving_pairs.sum()), turns=int(turned.sum()), per_stage=per_stage)


def compare(archive):
    baseline = load_run(archive, "baseline")
    variant = load_run(archive, "idle025")
    common = min(len(baseline["actions"]), len(variant["actions"]))
    difference = np.flatnonzero(baseline["actions"][:common] != variant["actions"][:common])
    if not len(difference):
        raise ValueError("The archived runs have no action split")
    i = int(difference[0])
    prefix = slice(None, i + 1)
    for field in ("features", "frames", "stages", "stage_frames", "position"):
        if not np.array_equal(baseline[field][prefix], variant[field][prefix]):
            raise ValueError(f"The runs differ in {field} before the first action split")
    # The inputs and all previous actions are still paired through this decision.
    # Once the histories diverge, each run needs its own observations and recurrent state.
    adjusted = apply_idle_bonus(baseline["logits"][prefix], baseline["features"][prefix])
    np.testing.assert_allclose(adjusted, variant["logits"][prefix], rtol=0, atol=3e-5)
    return dict(baseline=count_run(baseline), idle025=count_run(variant), first_action_difference=dict(
        decision_index=i, stage=int(variant["stages"][i]), stage_frame=int(variant["stage_frames"][i]),
        episode_frame=int(variant["frames"][i]), position=variant["position"][i].tolist(),
        baseline_action=int(baseline["actions"][i]), idle025_action=int(variant["actions"][i])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", help="Path to seed60036-traces.zip from the Release")
    options = parser.parse_args()
    print(json.dumps(compare(options.traces), indent=2))
