"""Translate the public choices into the existing original-game runners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from .run_contract import contract

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = {
    ("th14", "clear"): "connectome-compact129-full-story-v1",
    ("th14", "nbnm"): "connectome-nbnm-route407-corridor-v1",
    ("th08", "nbnm"): "connectome-nbnm-route407-corridor-v1",
}
DEFAULT_SEED = {("th14", "clear"): 3574003864, ("th14", "nbnm"): 60036, ("th08", "nbnm"): 18233}
STOCK_WEIGHT = {
    ("th14", "clear"): "th14-clear-cycle087.pt",
    ("th14", "nbnm"): "th14-nbnm-cycle073.pt",
    ("th08", "nbnm"): "th08-nbnm-18233.pt",
}


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def path_at(root, value):
    return (root / Path(value).expanduser()).resolve()


@dataclass
class RunPlan:
    operation: str
    root: Path
    output: Path
    controls: Path
    runner: Path
    arguments: list[str]
    settings: dict[Path, dict]
    inputs: tuple[Path, ...]
    identity: dict
    resume: Path | None = None

    def describe(self):
        return dict(operation=self.operation, identity=self.identity, output=str(self.output),
                    job=str(self.controls), runner=str(self.runner), arguments=self.arguments,
                    inputs=[str(path) for path in self.inputs], settings={str(k): v for k, v in self.settings.items()},
                    resume=str(self.resume) if self.resume else None)


def build_plan(options, *, root=ROOT):
    root = Path(root).resolve()
    game, objective = options.game, options.objective
    key = game, objective
    if key not in SUPPORT:
        raise ValueError("TH08 clear is pending. TH14 supports clear and nbnm.")
    if options.difficulty != "lunatic":
        raise ValueError("The supplied Windows profiles and model presets support Lunatic")
    if options.operation == "fit-attack" and key != ("th08", "nbnm"):
        raise ValueError("Published attack-local fine-tuning uses the original TH08 NBNM readout")
    if options.operation != "train" and options.resume is not None:
        raise ValueError("Only training accepts an optimizer checkpoint")

    call_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    jobs = path_at(root, options.jobs or "outputs/control/jobs")
    controls = jobs / f"{call_id}-{game}-{objective}"
    resume = path_at(root, options.resume) if options.resume else None
    output = (path_at(root, options.output) if options.output else
              resume.parent if resume and objective == "nbnm" else
              root / "outputs" / "workbench" / controls.name)
    settings_dir = output.with_name(output.name + ".settings")
    weights = path_at(root, options.checkpoint or "weights/published/" + STOCK_WEIGHT[key])
    connectome = path_at(root, options.connectome or "data/connectome")
    scope = options.train_scope if options.operation == "train" else "joint"
    identity = contract(game, options.difficulty, objective, scope, SUPPORT[key], heads=options.trainable_heads)
    if options.operation == "fit-attack":
        identity.update(train_scope="attack-readout", optimizer_groups=["native_attack_readout"])
    inputs = [weights, connectome]
    files = {}
    seed = DEFAULT_SEED[key] if options.seed is None else options.seed
    if not 0 <= seed < 2**(16 if game == "th08" else 32):
        raise ValueError(f"Seed outside the native range for {game}")

    if objective == "nbnm":
        spec = read_json(path_at(root, options.spec or f"configs/{game}/nbnm.json"))
        spec["connectome"] = str(connectome)
        spec["initial_checkpoint"] = str(weights)
        spec["output"] = str(output)
        spec_file = settings_dir / "spec.json"
        files[spec_file] = spec
        if options.operation in ("train", "fit-attack"):
            spec["run_contract"] = identity
        if options.operation == "train" and options.seed is not None:
            spec["seed"] = seed

    if options.operation == "fit-attack":
        guide = path_at(root, options.guide)
        runner = "fit_th08_attack_specialist.py"
        args = ["--spec", spec_file, "--checkpoint", weights, "--guide", guide,
                "--steps", options.fit_steps, "--output", output]
        inputs.extend((guide / "report.json", guide / "matched-history.npz"))
    else:
        if not options.executable:
            raise ValueError("Supply an original executable with --game-exe")
        executable = path_at(root, options.executable)
        inputs.append(executable)
        if game == "th08":
            adapter = path_at(root, options.d3d8to9 or "native/th08/d3d8.dll")
            profile = path_at(root, options.profile or "configs/th08-1.00d-zh-input.json")
            inputs.extend((adapter, profile, executable.parent / "th08.dat", executable.parent / "thbgm.dat"))
            args = ["--spec", spec_file, "--checkpoint", weights, "--executable", executable,
                    "--profile", profile, "--d3d8to9", adapter, "--output", output, "--unpaced"]
            if options.operation == "train":
                runner = "train_th08_nbnm.py"
                workers = options.workers or spec.get("workers", 8)
                segment = options.segment or spec.get("segment_decisions", 2048)
                decisions = options.decisions or spec.get("decision_budget", 32768)
                _validate_schedule(workers, segment, decisions)
                args += ["--train-scope", scope, "--workers", workers, "--segment-decisions", segment,
                         "--decisions", decisions, "--execute"]
                if options.course:
                    course = path_at(root, options.course)
                    args += ["--course", course]
                    inputs.append(course)
                if resume:
                    if resume == output / "latest-training.pt":
                        args += ["--resume"]
                    else:
                        args += ["--continue-from", resume]
            else:
                runner = "evaluate_th08_nbnm.py"
                args += ["--native-seed", seed, "--episodes", 1]
                if options.record_observations:
                    args.append("--record-observations")
        else:
            environment_file = settings_dir / "environment.json"
            source_env = options.environment or (spec["environment_config"] if objective == "nbnm"
                                                   else "configs/th14/clear-environment.json")
            environment = read_json(path_at(root, source_env))
            profile = path_at(root, options.profile or environment["profile"])
            environment.update(game_exe=str(executable), profile=str(profile))
            files[environment_file] = environment
            inputs.extend((profile, root / "native/th14_fake_d3d9.dll"))
            if objective == "nbnm":
                spec["environment_config"] = str(environment_file)
                spec["full_story_evaluation_seeds"] = [seed]
                if options.operation == "train":
                    runner = "train_nbnm.py"
                    workers = options.workers or spec["workers"]
                    segment = options.segment or spec["segment_decisions"]
                    decisions = options.decisions or spec["decision_budget"]
                    _validate_schedule(workers, segment, decisions)
                    groups = spec["world_groups"]
                    if len(groups) != workers:
                        if len(set(groups)) != 1:
                            raise ValueError("Set world_groups explicitly when resizing a mixed curriculum")
                        groups = [groups[0]] * workers
                    spec.update(workers=workers, world_groups=groups, segment_decisions=segment,
                                decision_budget=decisions, evaluate_at_end=False)
                    spec.pop("warmstart_optimizer_checkpoint", None)
                    args = ["--config", spec_file, "--execute"]
                    if resume:
                        args += ["--resume", resume]
                else:
                    runner = "evaluate_nbnm_checkpoint.py"
                    args = ["--config", spec_file, "--checkpoint", weights, "--output", output, "--task", "full_story"]
            else:
                args = ["--checkpoint", weights, "--connectome", connectome,
                        "--environment-config", environment_file, "--output", output]
                if options.operation == "train":
                    runner = "train_compact_story_ppo.py"
                    environment["run_contract"] = identity
                    workers = options.workers or 8
                    steps = options.segment or 512
                    if not 1 <= workers <= 30 or steps % 16:
                        raise ValueError("TH14 clear needs 1–30 worlds and a per-world segment divisible by 16")
                    decisions = options.decisions or 2 * steps * workers
                    if decisions <= 0 or decisions % (workers * steps):
                        raise ValueError("TH14 clear needs a whole number of workers × segment decisions")
                    args += ["--train-scope", scope, "--workers", workers, "--steps", steps,
                             "--cycles", decisions // (workers * steps), "--evaluation-episodes", 0,
                             "--defer-final-evaluation", "--execute"]
                    if resume:
                        args += ["--resume-training", resume]
                    if options.seed is not None:
                        args += ["--seed", seed]
                else:
                    runner = "evaluate_compact_repaired_checkpoint.py"
                    args += ["--kernel-cache", root / "outputs/cache/native-kernels", "--episodes", 1,
                             "--evaluation-seed", 2027432926, "--policy-seed", 2027432927,
                             "--action-selection", "family_mode", "--opening-seeds", seed]
    if resume:
        inputs.append(resume)
    if key == ("th14", "clear"):
        args = [options.operation, *args]
        runner = "run_public_th14_clear.py"
    runner = root / "scripts" / runner
    return RunPlan(options.operation, root, output, controls, runner, list(map(str, args)),
                   files, tuple(dict.fromkeys(inputs)), identity, resume)


def _validate_schedule(workers, segment, budget):
    if not 1 <= workers <= 100 or segment <= 0 or segment % 256 or budget <= 0 or budget % segment:
        raise ValueError("NBNM needs 1–100 workers and a positive budget in whole 256-row segments")


def windowless_python():
    if os.name != "nt":
        raise ValueError("Original game jobs require Windows")
    path = Path(sys.executable).with_name("pythonw.exe")
    if not path.is_file():
        raise FileNotFoundError(f"Install the Windows Python GUI executable beside {sys.executable}")
    return path


def launch(plan):
    from .thrl_durable_records import write_json
    from .public_jobs import list_jobs

    python = windowless_python()
    if not plan.runner.is_file():
        raise FileNotFoundError("Use an editable checkout containing scripts/ and configs/")
    missing = [path for path in plan.inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime inputs: " + ", ".join(map(str, missing)))
    if plan.output.exists() and (plan.resume is None or plan.identity["objective"] == "clear"):
        raise FileExistsError(f"Choose a new output directory: {plan.output}")
    active = list_jobs(plan.controls.parent, active_only=True)
    if any(Path(job["output"]).resolve() == plan.output for job in active):
        raise ValueError("A managed runner already uses this output directory")
    for path, data in plan.settings.items():
        if path.exists() and read_json(path) != data:
            raise ValueError(f"This output already has different engine settings: {path}")
    plan.controls.mkdir(parents=True)
    for path, data in plan.settings.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, data)
    if plan.resume:
        (plan.output / "STOP").unlink(missing_ok=True)
    local_temp = plan.controls / "temp"
    local_temp.mkdir()
    environment = dict(PYTHONPATH=os.pathsep.join(str(plan.root / path) for path in ("src", "scripts")),
                       PYTHONUTF8="1", FLY_TOUHOU_DESKTOP_NAME=f"fly-touhou-job-{plan.controls.name}",
                       TMP=str(local_temp), TEMP=str(local_temp), TMPDIR=str(local_temp))
    job = dict(schema="fly-touhou-background-job-v1", python=str(python), script=str(plan.runner),
               cwd=str(plan.root), control_directory=str(plan.controls), arguments=plan.arguments,
               environment=environment)
    record = plan.controls / "job.json"
    write_json(record, plan.describe())
    spec_file = plan.controls / "launch-spec.json"
    write_json(spec_file, job)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    with (plan.controls / "launcher.log").open("wb") as log:
        subprocess.Popen([str(python), str(plan.root / "scripts/background_job.py"), "--spec", str(spec_file)],
                         cwd=plan.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
    return record
