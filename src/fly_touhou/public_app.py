"""Command line for original-game runs, resumption, and the native console."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .training.public_runs import ROOT, SUPPORT, build_plan, launch


def positive(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("Choose a positive integer")
    return number


def add_job_options(parser):
    parser.add_argument("--jobs", type=Path, help="Directory containing managed job records")


def add_engine_options(parser, operation):
    parser.add_argument("--game", choices=("th14", "th08"), required=True)
    parser.add_argument("--difficulty", choices=("easy", "normal", "hard", "lunatic"), default="lunatic")
    parser.add_argument("--objective", choices=("clear", "nbnm"), required=True)
    parser.add_argument("--game-exe", "--executable", dest="executable", type=Path,
                        required=operation != "fit-attack")
    for key in ("checkpoint", "connectome", "output", "spec", "environment", "profile", "d3d8to9"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--seed", type=int, help="Native seed for evaluation or learner RNG seed for training")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact backend and settings without starting it")
    add_job_options(parser)
    parser.set_defaults(train_scope="joint", trainable_heads=None, resume=None, workers=None, decisions=None,
                        segment=None, course=None, guide=None, fit_steps=640, record_observations=False)
    if operation == "train":
        parser.add_argument("--train-scope", choices=("joint", "encoder-heads"), default="joint")
        parser.add_argument("--trainable-head", action="append", dest="trainable_heads",
                            help="Repeat for each output group to optimize")
        parser.add_argument("--resume", type=Path, help="Same-contract optimizer checkpoint")
        parser.add_argument("--workers", type=positive)
        parser.add_argument("--decisions", type=positive)
        parser.add_argument("--segment", type=positive, help="TH14 clear uses per-world decisions")
        parser.add_argument("--course", type=Path, help="Original TH08 course manifest")
    elif operation == "fit-attack":
        parser.add_argument("--guide", type=Path, required=True,
                            help="Complete original-game no-contact certificate and matched-history.npz")
        parser.add_argument("--fit-steps", type=positive, default=640)
    else:
        parser.add_argument("--record-observations", action="store_true", help="TH08 only")


def parser_for_app():
    parser = argparse.ArgumentParser(prog="fly-touhou", description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for command, description in (
        ("train", "Start or resume original-game training"),
        ("evaluate", "Evaluate a frozen policy from the original Story opening"),
        ("fit-attack", "Fit the TH08 readout for a single verified original attack"),
    ):
        add_engine_options(sub.add_parser(command, help=description), command)
    add_job_options(sub.add_parser("console", help="Open the local Windows job console"))
    status = sub.add_parser("status", help="Read background jobs and their latest progress")
    add_job_options(status)
    status.add_argument("--active", action="store_true")
    stop = sub.add_parser("stop", help="Request a safe stop at the next supported boundary")
    stop.add_argument("job", type=Path, help="Job directory or job.json path")
    sub.add_parser("support", help="List the published game, difficulty and objective combinations")
    legacy = sub.add_parser("legacy", help="Use an earlier command without changing its arguments")
    legacy.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and (argv[0] in ("doctor", "launch", "inspect", "collect", "train-teacher", "distill", "train-student-ppo")
                 or argv[0] == "evaluate" and len(argv) > 1 and "--game" not in argv
                 and not any(arg.startswith("--game=") for arg in argv) and argv[1] not in ("--help", "-h")):
        from .cli import main as legacy
        return legacy(argv)
    parser = parser_for_app()
    args = parser.parse_args(argv)
    try:
        if args.operation == "legacy":
            from .cli import main as legacy
            return legacy(args.arguments)
        if args.operation == "support":
            result = [dict(game=game, difficulty="lunatic", objective=goal, observation_version=observation,
                           train_scopes=["joint", "encoder-heads"]) for (game, goal), observation in SUPPORT.items()]
        elif args.operation == "console":
            from .training.console import open_console
            return open_console((ROOT / (args.jobs or "outputs/control/jobs")).resolve())
        elif args.operation == "status":
            from .training.public_jobs import list_jobs
            result = list_jobs((ROOT / (args.jobs or "outputs/control/jobs")).resolve(), active_only=args.active)
        elif args.operation == "stop":
            from .training.public_jobs import request_stop
            result = {"stop_request": str(request_stop(args.job))}
        else:
            plan = build_plan(args)
            result = plan.describe() if args.dry_run else {"job": str(launch(plan))}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


def console_main():
    return main(["console"])


if __name__ == "__main__":
    raise SystemExit(main())
