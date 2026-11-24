"""Offline checks of the consolidated entry points and forecast evidence."""

import json

import pytest

from fly_touhou.games.contracts import Coverage, Movement
from fly_touhou.games.th08.adapter import TH08Adapter
from fly_touhou.public_app import parser_for_app
from fly_touhou.training.public_runs import build_plan
from fly_touhou.training.run_contract import contract, restore_contract


def plan(command, game, objective, *options):
    args = ["evaluate" if command == "evaluate" else "train", "--game", game,
            "--objective", objective, "--game-exe", "D:/owned game/original.exe", *options]
    return build_plan(parser_for_app().parse_args(args))


@pytest.mark.parametrize(("game", "objective", "train", "frozen"), (
    ("th14", "clear", "run_public_th14_clear.py", "run_public_th14_clear.py"),
    ("th14", "nbnm", "train_nbnm.py", "evaluate_nbnm_checkpoint.py"),
    ("th08", "nbnm", "train_th08_nbnm.py", "evaluate_th08_nbnm.py"),
))
def test_published_runner_selection(game, objective, train, frozen):
    training = plan("train", game, objective, "--train-scope", "encoder-heads")
    evaluation = plan("evaluate", game, objective)
    assert training.runner.name == train and training.runner.is_file()
    assert evaluation.runner.name == frozen and evaluation.runner.is_file()
    assert training.identity["train_scope"] == "encoder-heads"
    assert "brain" not in training.identity["optimizer_groups"]
    assert any("owned game" in str(path) for path in evaluation.inputs)


def test_th14_nbnm_example_matches_existing_training_schema(tmp_path):
    from fly_touhou.nbnm_run import validate_spec

    prepared = plan("train", "th14", "nbnm", "--train-scope", "joint")
    settings = {path.name: data for path, data in prepared.settings.items()}
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps(settings["environment.json"]), encoding="utf-8")
    spec = dict(settings["spec.json"], environment_config=str(environment))
    assert validate_spec(spec)["status"] == "ready"


def test_scope_switch_cannot_restore_old_adam():
    before = contract("th14", "lunatic", "clear", "joint", "features-v1")
    after = contract("th14", "lunatic", "clear", "encoder-heads", "features-v1")
    restore_contract({"model_payload": {"run_contract": before}}, before)
    with pytest.raises(ValueError, match="Optimizer resume"):
        restore_contract({"model_payload": {"run_contract": before}}, after)


def test_truncated_original_trace_has_unknown_tail():
    tape = (Movement(1, 12),)
    trace = [{"geometry_valid": True, "clearance": 2., "contact": False}] * 4
    trace.append({"geometry_valid": False, "clearance": None, "contact": False})
    route = TH08Adapter._route(tape, trace, (4, 12))
    assert route.verified_corridor_frames == 4
    assert route.horizons[0].coverage is Coverage.ORIGINAL and route.horizons[0].contact_free is True
    assert route.horizons[1].coverage is Coverage.UNKNOWN and route.horizons[1].contact_free is None
