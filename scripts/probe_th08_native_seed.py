"""Prove native TH08 startup-seed reproducibility before paired policy tests."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.th08_controller import TH08InputController
from fly_touhou.th08_training import training_worlds
from fly_touhou.thrl_durable_records import write_json


def selected(table, columns):
    return {name: table[name].tolist() for name in columns}


def measure(args, directory, seed):
    world = training_worlds(args.executable, args.profile, args.d3d8to9, directory, 1, {})[0]
    with TH08InputController(world["executable"], args.profile, private_appdata=world["private_appdata"],
                            d3d8to9=args.d3d8to9, unpaced=True, native_seed=seed) as controller:
        state = controller.enter_story(3)
        observed = state.get("native_seed")
        if not observed or observed["requested"] != seed or observed["applied"] != 1:
            raise RuntimeError("The original startup seed was not observed")
        start = state["input_poll"]
        captured = []
        while True:
            now = state["input_poll"] - start
            game = controller.read_game_state(state)
            projectiles = controller.read_projectiles()
            enemies = controller.read_enemies()
            captured.append(dict(native_frames=now, stage_frame=state["stage_timer"], raw_stage=state["stage"],
                recorded_stage_seed=game["stage_rng_seed"], position=state["player_position"], time=game.get("clock"),
                power=game.get("power"), score_time_orbs=game.get("time_orbs"), contact=state["native_contact"],
                bullets=selected(projectiles["bullets"], ("slot", "position", "velocity", "state", "lifetime")),
                lasers=selected(projectiles["lasers"], ("slot", "origin", "angle", "elapsed", "state")),
                enemies=selected(enemies, ("slot", "position", "flags", "life", "script_clock"))))
            if now >= args.frames or state["mode"] != 2 or state["stage"] != 0:
                break
            state = controller.advance(1, min(16, args.frames - now))
    return dict(seed=seed, startup=observed, snapshots=captured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "d3d8to9", "output"):
        parser.add_argument(f"--{name}", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/th08-1.00d-zh-input.json")
    parser.add_argument("--native-seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.native_seed <= 65535 or args.frames < 512:
        parser.error("Use a 16-bit seed and at least 512 probe frames")
    with gpu_phase("evaluation"):
        args.output.mkdir(parents=True)
        with TemporaryDirectory(prefix="original-games-", dir=args.output) as directory:
            native = Path(directory).resolve(strict=True)
            if native.parent != args.output:
                raise ValueError("Keep the isolated original game under the probe output")
            capture = [
                measure(args, native / str(i), seed)
                for i, seed in enumerate((args.native_seed, args.native_seed, args.native_seed ^ 0x8000))
            ]
        for i, run in enumerate(capture):
            write_json(args.output / f"trace-{i}.json", run)
        a, b, other = [run["snapshots"] for run in capture]
        observable = any(row["bullets"]["slot"] for row in a)
        recorded_seeds = [x[0]["recorded_stage_seed"] for x in (a, b, other)]
        same = a == b
        reached_budget = all(rows[-1]["native_frames"] == args.frames for rows in (a, b, other))
        passed = same and reached_budget and observable and a[0]["raw_stage"] == 0 and recorded_seeds[0] != recorded_seeds[2]
        write_json(args.output / "report.json", dict(paired_seed=args.native_seed, different_seed=args.native_seed ^ 0x8000,
            exact_native_snapshots=same, observed_projectiles=observable, recorded_stage_seeds=recorded_seeds,
            original_frames=[x[-1]["native_frames"] for x in (a,b,other)], snapshots=[len(x) for x in (a,b,other)],
            native_audio=False, seed_parity_verified=passed, full_game_NBNM=False))
        if not passed:
            raise SystemExit("The original-game paired seed probe did not verify state parity")


if __name__ == "__main__":
    main()
