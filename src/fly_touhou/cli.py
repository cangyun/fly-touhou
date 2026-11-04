"""Background-only original-game experiments with explicit native evidence."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "games/th14_trial/extracted/app/th14.exe"
DEFAULT_PROFILE = ROOT / "profiles/th14_trial_0.01b.json"
DEFAULT_DATA = ROOT / "data/connectome"


def write_report(path, report, *, summary=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    visible = summary if summary is not None else report
    print(json.dumps({"report": str(path.resolve()), **visible}, ensure_ascii=False, indent=2, default=str), flush=True)


def choose_target(args):
    """Read-only inspection may select a pre-existing original-game process."""
    from .native import NativeError, Profile, find_game
    profile = Profile.load(args.profile)
    if args.pid is not None:
        return args.pid, profile
    found = find_game(profile.executable_names)
    if len(found) != 1:
        raise NativeError(f"Expected one read-only inspection target; found {found}. Select it with --pid.")
    return found[0]["pid"], profile


def doctor(args):
    import platform
    import torch
    from .native import Profile, find_game, sha256_file
    profiles = []
    for path in (ROOT / "profiles").glob("*.json"):
        definition = json.loads(path.read_text(encoding="utf-8"))
        profiles.append({"path": str(path), "name": definition["name"],
                         "verification": definition.get("verification"),
                         "pinned_executables": len(definition["sha256"])})
    dependencies = {}
    for name in ("frida", "pywin32", "stable-baselines3", "gymnasium"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    game = {"path": str(DEFAULT_EXE), "exists": DEFAULT_EXE.is_file()}
    if game["exists"]:
        game["sha256"] = sha256_file(DEFAULT_EXE)
        game["pinned"] = game["sha256"] in Profile.load(DEFAULT_PROFILE).sha256
    report = {"platform": platform.platform(), "python": sys.executable,
              "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "native_only": True, "runtime_backend": "original-th14-frida-ipc",
              "global_desktop_input": False, "browser_backend": False,
              "default_game": game, "dependencies": dependencies,
              "processes": find_game([f"th{g}.exe" for g in (7, 14, 16)]), "profiles": profiles}
    write_report(args.output or ROOT / "artifacts/doctor.json", report)


def launch(args):
    """Probe our owned original EXE and close it before returning."""
    from .frida_controller import OriginalGameController
    controller = OriginalGameController(args.exe, args.profile, timeout=args.timeout)
    try:
        controller.start()
        controller.ensure_run()
        snapshot = controller.read()
        report = {"source": snapshot.source, "exe": str(Path(args.exe).resolve()),
                  "exe_sha256": controller.reader.digest, "pid": controller.pid,
                  "profile": controller.profile.name, "snapshot": snapshot.to_dict(),
                  "closed_after_probe": True}
    finally:
        controller.close()
    write_report(args.output or ROOT / "artifacts/background_probe.json", report,
                 summary={"source": report["source"], "exe_sha256": report["exe_sha256"],
                          "backend_kind": snapshot.extra.get("backend_kind"),
                          "stage": snapshot.stage, "stage_frame": snapshot.stage_frame,
                          "player": snapshot.player, "closed_after_probe": True})


def inspect(args):
    """Only read process memory: no screenshot, input or focus operations."""
    from .native import NativeReader, ProcessMemory
    if args.samples < 1 or args.interval < 0:
        raise ValueError("samples must be positive and interval nonnegative")
    pid, profile = choose_target(args)
    with ProcessMemory(pid) as memory:
        reader = NativeReader(memory, profile, inspect_unverified=args.inspect_unverified)
        snapshots = []
        for index in range(args.samples):
            snapshots.append(reader.read().to_dict())
            if index + 1 < args.samples:
                time.sleep(args.interval)
    report = {"profile": profile.name, "pid": pid, "exe_sha256": reader.digest,
              "identity_pinned": reader.verified_identity, "source": "native-process",
              "access": "read-only", "snapshots": snapshots}
    write_report(args.output or ROOT / "artifacts/native_probe.json", report,
                 summary={key: value for key, value in report.items() if key != "snapshots"})


def environment(args):
    from .background import BackgroundTouhouEnv
    from .observation import ObservationConfig
    return BackgroundTouhouEnv(args.exe, args.profile, frame_skip=args.frame_skip,
        max_episode_frames=args.episode_frames, timeout=args.timeout,
        observation_config=ObservationConfig(include_action_risk=not args.raw))


def collect(args):
    from .learning import collect_demonstrations
    env = environment(args)
    try:
        if args.teacher:
            from stable_baselines3 import PPO
            teacher_arguments = {"teacher": PPO.load(args.teacher, device="cpu")}
        else:
            teacher_arguments = {"planner": env.encoder.planner}
        report = collect_demonstrations(env, args.output, episodes=args.episodes,
            encoder=env.encoder, max_steps=args.max_steps, seed=args.seed,
            deterministic=not args.sample_actions, **teacher_arguments)
        report_path = Path(args.output).with_suffix(".report.json")
        write_report(report_path, report, summary={key: report[key] for key in (
            "teacher_kind", "total_steps", "total_native_frames", "decisions_with_hazards",
            "max_collidable_bullets", "max_active_lasers", "samples_per_action")})
    finally:
        env.close()


def train_teacher(args):
    from .learning import train_ppo_teacher
    env = environment(args)
    try:
        _, report = train_ppo_teacher(env, args.output, total_timesteps=args.steps,
            seed=args.seed, n_steps=args.rollout_steps, batch_size=args.batch_size, device=args.device)
        write_report(Path(report["checkpoint"]).with_suffix(".report.json"), report,
            summary={key: report[key] for key in ("algorithm", "initialization", "actual_timesteps",
                                                  "native_steps", "native_frames", "checkpoint")})
    finally:
        env.close()


def distill(args):
    from .connectome import load_connectome
    from .learning import distill_student
    if args.neurons < 0:
        raise ValueError("neurons must be positive; use zero for the full available graph")
    graph = load_connectome(args.data, max_neurons=args.neurons or None)
    report = distill_student(args.traces, graph, args.output, epochs=args.epochs,
        chunk_size=args.chunk_size, device=args.device, seed=args.seed,
        learning_rate=args.learning_rate, validation_fraction=args.validation_fraction)
    write_report(Path(args.output).with_suffix(".report.json"), report,
        summary={key: report[key] for key in ("training", "epochs", "training_metrics",
            "validation_metrics", "checkpoint", "evidence_scope")})


def restore_student(args):
    """A checkpoint's exact source indices determine its graph, never a new size."""
    import torch
    from .connectome import load_connectome
    from .learning import STUDENT_SCHEMA, load_student_checkpoint
    if not args.checkpoint:
        raise ValueError("Student operation requires --checkpoint")
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != STUDENT_SCHEMA or "graph_source_indices" not in checkpoint:
        raise ValueError("Student checkpoint lacks the exact saved connectome neuron indices")
    source_indices = checkpoint["graph_source_indices"]
    del checkpoint
    graph = load_connectome(args.data, source_indices=source_indices)
    return load_student_checkpoint(args.checkpoint, graph, device=args.device)


def train_student(args):
    from .student_ppo import train_student_ppo
    policy = restore_student(args)
    env = environment(args)
    try:
        report = train_student_ppo(env, policy, args.output, total_timesteps=args.steps,
            rollout_steps=args.rollout_steps, sequence_length=args.sequence_length,
            n_epochs=args.ppo_epochs, minibatch_chunks=args.minibatch_chunks,
            learning_rate=args.learning_rate, seed=args.seed,
            resume_checkpoint=args.checkpoint if args.resume else None)
        write_report(Path(args.output).with_suffix(".report.json"), report,
            summary={key: report[key] for key in ("training", "native_steps", "native_frames",
                "decisions_with_hazards", "max_collidable_bullets", "checkpoint", "evidence_scope")})
    finally:
        env.close()


def evaluate(args):
    from .learning import evaluate_policy
    env = environment(args)
    try:
        if args.policy == "planner":
            policy = env.encoder.planner
        elif args.policy == "stay":
            policy = lambda observation: 0
        elif args.policy == "ppo":
            from stable_baselines3 import PPO
            if not args.checkpoint:
                raise ValueError("PPO evaluation requires --checkpoint")
            policy = PPO.load(args.checkpoint, device=args.device)
        else:
            policy = restore_student(args)
        report = evaluate_policy(env, policy, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps)
        report["policy_name"] = args.policy
        write_report(args.output, report,
            summary={key: report[key] for key in ("policy_name", "mean_native_frames", "mean_reward",
                "hit_rate", "stage_clear_rate", "decision_latency_ms", "evidence_scope")})
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="果蝇连接组 × 东方原作 EXE：后台运行与训练，不占用桌面输入")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="只读检查环境、GPU、连接组及原作文件")
    d.add_argument("--output")
    d.set_defaults(func=doctor)

    def original_target(command):
        command.add_argument("--exe", default=str(DEFAULT_EXE), help="受版本哈希约束的东方原作 EXE")
        command.add_argument("--profile", default=str(DEFAULT_PROFILE))
        command.add_argument("--timeout", type=float, default=15)

    def runtime(command):
        original_target(command)
        command.add_argument("--frame-skip", type=int, default=4)
        command.add_argument("--episode-frames", type=int, default=5400)
        command.add_argument("--raw", action="store_true", help="关闭人工轨迹预测特征；须与训练配置一致")
        command.add_argument("--seed", type=int, default=0, help="训练随机种子，不代表已控制原作 RNG")

    def checkpoint_target(command, *, required=False):
        command.add_argument("--checkpoint", required=required)
        command.add_argument("--data", default=str(DEFAULT_DATA))
        command.add_argument("--device", default="cpu")

    l = sub.add_parser("launch", help="短暂启动原作、读取验证状态并关闭，不发送桌面按键")
    original_target(l)
    l.add_argument("--output")
    l.set_defaults(func=launch)

    i = sub.add_parser("inspect", help="只读检查已有原作进程；不操作窗口或截图")
    i.add_argument("--profile", default=str(DEFAULT_PROFILE))
    i.add_argument("--pid", type=int)
    i.add_argument("--inspect-unverified", action="store_true")
    i.add_argument("--samples", type=int, default=1)
    i.add_argument("--interval", type=float, default=.1)
    i.add_argument("--output")
    i.set_defaults(func=inspect)

    c = sub.add_parser("collect", help="通过进程内接口，在后台原作 EXE 中采集教师示范")
    runtime(c)
    c.add_argument("--teacher", help="PPO 教师检查点；省略则使用预测规划教师")
    c.add_argument("--sample-actions", action="store_true", help="从教师分布采样；默认选概率最大动作")
    c.add_argument("--episodes", type=int, default=8)
    c.add_argument("--max-steps", type=int, default=10000)
    c.add_argument("--output", required=True)
    c.set_defaults(func=collect)

    t = sub.add_parser("train-teacher", help="在后台原作 EXE 中从随机权重训练 PPO 教师")
    runtime(t)
    t.add_argument("--steps", type=int, default=8192)
    t.add_argument("--rollout-steps", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--device", default="cpu")
    t.add_argument("--output", required=True)
    t.set_defaults(func=train_teacher)

    s = sub.add_parser("distill", help="用原作真实轨迹离线蒸馏连接组学生；无需启动游戏")
    s.add_argument("--traces", nargs="+", required=True)
    s.add_argument("--data", default=str(DEFAULT_DATA))
    s.add_argument("--neurons", type=int, default=2048, help="子图神经元数；0 为可用完整连接组")
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--chunk-size", type=int, default=32)
    s.add_argument("--learning-rate", type=float, default=3e-4)
    s.add_argument("--validation-fraction", type=float, default=.2)
    s.add_argument("--device", default="cuda")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--output", required=True)
    s.set_defaults(func=distill)

    r = sub.add_parser("train-student-ppo", help="从蒸馏学生继续在后台原作中进行循环 PPO 训练")
    runtime(r)
    checkpoint_target(r, required=True)
    r.add_argument("--steps", type=int, default=8192)
    r.add_argument("--rollout-steps", type=int, default=128)
    r.add_argument("--sequence-length", type=int, default=16)
    r.add_argument("--ppo-epochs", type=int, default=4)
    r.add_argument("--minibatch-chunks", type=int, default=4)
    r.add_argument("--learning-rate", type=float, default=1e-4)
    r.add_argument("--resume", action="store_true", help="同时恢复检查点中的 PPO 价值网络与优化器")
    r.add_argument("--output", required=True)
    r.set_defaults(func=train_student)

    e = sub.add_parser("evaluate", help="在后台原作 EXE 中独立评测策略")
    runtime(e)
    checkpoint_target(e)
    e.add_argument("--policy", choices=("stay", "planner", "ppo", "student"), default="planner")
    e.add_argument("--episodes", type=int, default=5)
    e.add_argument("--max-steps", type=int, default=10000)
    e.add_argument("--output", required=True)
    e.set_defaults(func=evaluate)
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("已停止；本次后台原作子进程已关闭。", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
