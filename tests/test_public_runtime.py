"""Offline checks of the consolidated entry points and forecast evidence."""

from dataclasses import replace
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_launch_creates_settings_before_dispatch(tmp_path, monkeypatch):
    from fly_touhou.training import public_runs

    settings_path = tmp_path / "new-run.settings" / "spec.json"
    prepared = replace(plan("train", "th08", "nbnm"),
                       output=tmp_path / "new-run", controls=tmp_path / "jobs" / "job-1",
                       inputs=(), settings={settings_path: {"workers": 8}})
    calls = []
    monkeypatch.setattr(public_runs, "windowless_python", lambda: Path(sys.executable))
    monkeypatch.setattr(public_runs.subprocess, "STARTUPINFO", lambda: SimpleNamespace(dwFlags=0), raising=False)
    monkeypatch.setattr(public_runs.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr(public_runs.subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    monkeypatch.setattr(public_runs.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    record = public_runs.launch(prepared)
    assert record.is_file() and len(calls) == 1
    assert public_runs.read_json(settings_path) == {"workers": 8}
    assert not prepared.output.exists()


def test_web_console_lists_plans_and_stops_managed_jobs(tmp_path):
    from threading import Thread
    from urllib.error import HTTPError
    from urllib.request import ProxyHandler, Request, build_opener

    from fly_touhou.training.console import create_console_server

    jobs = tmp_path / "jobs"
    record = jobs / "sample-job"
    record.mkdir(parents=True)
    output = tmp_path / "sample-output"
    output.mkdir()
    (record / "job.json").write_text(json.dumps({
        "operation": "train", "output": str(output),
        "identity": {"game": "th08", "objective": "nbnm", "difficulty": "lunatic"},
    }))
    (record / "status.json").write_text('{"status":"running"}')
    (record / "stdout.log").write_text("rollout saved")
    (output / "progress.json").write_text('{"phase":"sampling","completed_decisions":1024,"optimizer_updates":12}')

    with create_console_server(jobs, 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            origin = f"http://127.0.0.1:{server.server_port}"
            opener = build_opener(ProxyHandler({}))

            def get(path):
                with opener.open(origin + path, timeout=3) as reply:
                    return json.load(reply)

            def post(path, body, *, request_origin=origin):
                data = json.dumps(body).encode()
                headers = {"Origin": request_origin, "X-Console-Token": token, "Content-Type": "application/json"}
                with opener.open(Request(origin + path, data=data, headers=headers), timeout=3) as reply:
                    return json.load(reply)

            token = get("/api/session")["token"]
            row, = get("/api/jobs")["jobs"]
            assert (row["id"], row["decisions"], row["updates"]) == ("sample-job", 1024, 12)
            assert get("/api/log?id=sample-job") == {"text": "rollout saved"}
            plan = post("/api/plan", {"game": "th08", "executable": "D:/Games/TH08/th08.exe"})
            assert plan["plan"]["runner"].endswith("train_th08_nbnm.py")
            with pytest.raises(HTTPError) as wrong_origin:
                post("/api/stop", {"id": "sample-job"}, request_origin="http://example.test")
            assert wrong_origin.value.code == 403 and not (output / "STOP").exists()
            post("/api/stop", {"id": "sample-job"})
            assert (output / "STOP").is_file() and get("/api/jobs")["jobs"][0]["status"] == "stopping"
        finally:
            server.shutdown()
            thread.join()
