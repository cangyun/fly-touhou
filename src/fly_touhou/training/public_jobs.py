"""Read and manage only the jobs dispatched by the public launcher."""

from pathlib import Path

from ..thrl_study_io import read_shared_json
from .public_runs import read_json

ACTIVE = {"starting", "running", "stopping"}


def _read_when_present(path):
    try:
        return read_shared_json(path)
    except (FileNotFoundError, PermissionError):
        return {}


def job_summary(record):
    control = record.parent
    plan = read_json(record)
    output = Path(plan["output"])
    status = _read_when_present(control / "status.json")
    progress = _read_when_present(output / "progress.json")
    failure = _read_when_present(output / "failure.json")
    report_path = output / "report.json"
    result = _read_when_present(report_path)
    if not result and plan["operation"] != "evaluate":
        report_path = output / "result.json"
        result = _read_when_present(report_path)
    info = progress.get("info") or {}
    if result.get("episodes") and isinstance(result["episodes"], list):
        if isinstance(result["episodes"][0], dict):
            info = result["episodes"][0]
    if isinstance(result.get("end"), dict):
        info = result["end"]
    status_name = status.get("status", "starting")
    if failure.get("operator_requested_stop") or result.get("operator_requested_stop") or progress.get("phase") == "stopped":
        status_name = "stopped"
    elif failure:
        status_name = "failed"
    elif (output / "STOP").exists() and status_name == "running":
        status_name = "stopping"
    if status_name in ACTIVE and progress.get("PID") is not None and progress.get("create_time") is not None:
        import psutil
        try:
            worker = psutil.Process(progress["PID"])
            owned = worker.is_running() and abs(worker.create_time() - progress["create_time"]) < 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            owned = False
        if not owned and not (result or failure):
            status_name = "exited"
    decisions = progress.get("completed_decisions", progress.get("new_student_actions", progress.get("decisions")))
    if decisions is None:
        decisions = info.get("policy_decisions", result.get("sampled_decisions",
                                                               result.get("autonomous_local_decisions")))
    updates = progress.get("optimizer_updates", progress.get("PPO_updates", progress.get("updates")))
    return dict(job=str(record), operation=plan["operation"], **plan["identity"],
                status=status_name, phase=progress.get("phase"), elapsed_seconds=progress.get("elapsed_seconds"),
                stage=info.get("stage", progress.get("stage")), frame=info.get("stage_frame", progress.get("stage_frame")),
                decisions=decisions, budget=progress.get("budget"), updates=updates,
                matched=progress.get("matched"), route_decisions=progress.get("route_decisions"),
                miss=info.get("misses"), bomb=info.get("Bombs"), contact=info.get("contact"),
                outcome=info.get("outcome"),
                error=failure.get("error") or status.get("error"), output=str(output),
                report=str(report_path) if result else None)


def list_jobs(directory, *, active_only=False):
    jobs = (job_summary(record) for record in sorted(directory.glob("*/job.json"), reverse=True))
    return [job for job in jobs if not active_only or job["status"] in ACTIVE]


def request_stop(path):
    path = Path(path).resolve()
    record = path if path.name == "job.json" else path / "job.json"
    row = job_summary(record)
    output = Path(row["output"])
    if row["status"] not in ACTIVE or not output.is_dir():
        raise ValueError("This job has no active runner output")
    target = output / "STOP"
    target.touch(exist_ok=True)
    return target
