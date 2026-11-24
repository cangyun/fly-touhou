"""Inventory loaded retail TH08 stage scripts through the original Practice menu."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.native import NativeError
from fly_touhou.th08_controller import TH08InputController
from fly_touhou.th08_training import training_worlds
from fly_touhou.thrl_durable_records import write_json


def read_stage(world, slot, seed):
    with TH08InputController(world["executable"], world["profile"], private_appdata=world["private_appdata"],
                             d3d8to9=world["d3d8to9"], unpaced=True, native_seed=seed) as controller:
        controller.enter_stage_practice(3, slot)
        source = controller.read_forecast_source()
        program = source["ecl"]["program"]
        subroutines = Counter(row["opcode"] for row in program.instructions(timeline=False))
        timelines = Counter(row["opcode"] for row in program.instructions(timeline=True))
        return {"program_bytes": len(program.data), "subroutines": len(program.subroutines),
                "timelines": len(program.timelines), "subroutine_opcodes": dict(sorted(subroutines.items())),
                "timeline_opcodes": dict(sorted(timelines.items())), "active_emitters": len(source["ecl"]["emitters"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=27109)
    parser.add_argument("--stages", type=int, nargs="+", choices=range(8), default=list(range(8)))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    with gpu_phase("evaluation"), TemporaryDirectory(prefix="stage-catalog-", dir=args.output) as temp:
        world, = training_worlds(plan["executable"], plan["profile"], plan["d3d8to9"], temp, 1, {},
                                 practice_slots=(args.stages[0],))
        for slot in args.stages:
            try:
                records[slot] = read_stage(world, slot, args.seed)
            except NativeError as error:
                records[slot] = {"error": str(error)}
            print(slot, {key: value for key, value in records[slot].items() if not key.endswith("opcodes")},
                  flush=True)
            write_json(args.output / "report.json", {"native_seed": args.seed, "mode": "lunatic-stage-practice",
                                                     "all_requested_stages": args.stages, "stages": records})


if __name__ == "__main__":
    main()
