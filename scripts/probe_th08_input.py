"""Verify original TH08 input and frame counters in a private game copy."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.th08_audio import configure_private_audio
from fly_touhou.th08_controller import TH08InputController, th08_action_mask


def motion_audit(controller, result):
    moves = [(action, 4) for action in range(18)] + [(10, 60), (13, 60)]
    measured = []
    result["motion_audit"] = measured
    for action, ticks in moves:
        motion = controller.read_motion()
        mask = th08_action_mask(action)
        expected = motion.paths([mask], ticks)[0, -1].tolist()
        actual = controller.advance(mask, ticks)
        position = actual["player_position"]
        error = max(abs(a - b) for a, b in zip(expected, position))
        measured.append(dict(action=action, ticks=ticks, position=position, predicted=expected,
                             max_error=error, geometry=asdict(motion), stage_timer=actual["stage_timer"]))
        if error > 0.025:
            raise RuntimeError(f"TH08 action {action} differs from the native path by {error:.4f} pixels")


def step(value):
    mask, ticks = value.split(":", 1)
    return int(mask, 0), int(ticks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--private-game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d3d8to9", type=Path, required=True)
    startup = parser.add_mutually_exclusive_group()
    startup.add_argument("--step", type=step, action="append")
    startup.add_argument("--difficulty", type=int, choices=range(4))
    parser.add_argument("--verify-motion", action="store_true")
    args = parser.parse_args()

    original = args.game.resolve(strict=True)
    private = args.private_game.resolve()
    if private == original.parent:
        parser.error("Use a separate private game directory")
    private.mkdir(parents=True, exist_ok=True)
    files = (original.name, "th08.dat", "thbgm.dat", "score.dat")
    for name in files:
        destination = private / name
        if not destination.exists():
            shutil.copy2(original.parent / name, destination)
    configure_private_audio(original.parent / "th08.cfg", private / "th08.cfg")

    steps = args.step
    if steps is None and args.difficulty is None:
        steps = [(0, 136), (1, 1), (0, 60), (1, 1), (0, 36),
                 (1, 1), (0, 36), (1, 1), (0, 360)]
        steps.extend((th08_action_mask(action), 4) for action in (10, 11, 12, 13, 2, 1))
    profile = ROOT / "configs" / "th08-1.00d-zh-input.json"
    controller = TH08InputController(
        private / original.name, profile, private_appdata=private / "appdata", d3d8to9=args.d3d8to9)
    result = {"profile": controller.profile.name, "source": str(original),
              "private_game": str(private), "steps": [], "complete": False}
    try:
        with controller:
            before = controller.read()
            result["start"] = before
            print(json.dumps(before), flush=True)
            if args.difficulty is not None:
                before = controller.enter_story(args.difficulty)
                result["story_entry"] = before
            for mask, ticks in steps or ():
                after = controller.advance(mask, ticks)
                if after["input_poll"] - before["input_poll"] != ticks:
                    raise RuntimeError("TH08 consumed a different number of input polls")
                result["steps"].append({"mask": mask, "requested_polls": ticks, **after})
                print(json.dumps(result["steps"][-1]), flush=True)
                before = after
            if args.verify_motion:
                motion_audit(controller, result)
            result["complete"] = True
    finally:
        result["startup_trace"] = controller._milestones
        if controller._isolated_child is not None:
            result["desktop_isolation"] = controller._isolated_child.audit
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
