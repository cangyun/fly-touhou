"""Finite real-D95 CUDA qualification with verified main lifecycle continuation.

No native process, rollout, deployment checkpoint or production fit is made.
Two Adam steps affect disposable independent model copies only. Late archived
states are independent single-step numeric probes, not zero-history BC data.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now

D95 = Path("./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt")
D95_SHA = "d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c"
ARCHIVE = Path("./outputs/engineering/B1-autonomous-dodge-20261024/teacher-full-natural30-v1")
PRECISION = dict(parameter_dtype="float32", observation_dtype="float32", autocast_enabled=False,
                 matmul_allow_tf32=False, cudnn_allow_tf32=False)
CHECKS = ("final_wrapper_T1_parity", "full_encoder_gradients_and_changes",
          "control_encoder_unchanged", "critic_and_risk_unchanged",
          "own_hidden_block_continuation", "inventory_legal_CE_finite",
          "planned_batch_forward_backward")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preflight(expected_manifest):
    manifest = ROOT / "source_manifest.json"
    if not manifest.is_file() or sha(manifest) != expected_manifest:
        raise ValueError("A pinned, exclusive frozen runtime is required")
    info = read(manifest)
    pins = {}
    for row in info["files"]:
        relative = Path(row["path"])
        path = (ROOT / relative).resolve()
        if (relative.is_absolute() or not path.is_relative_to(ROOT.resolve())
                or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]):
            raise ValueError("Frozen source bytes differ: " + str(relative))
        if relative.as_posix().startswith("src/fly_touhou/") and relative.suffix == ".py":
            pins[relative.as_posix()] = row["sha256"]
    required = ("thrl_full_actor_path_bc.py", "thrl_full_actor_bc_stream.py", "thrl_full_teacher_data.py",
                "thrl_teacher_archive.py", "thrl_model.py", "thrl_pair_policy.py")
    if any("src/fly_touhou/" + name not in pins for name in required) or sha(D95) != D95_SHA:
        raise ValueError("Missing BC/NN source or wrong D95 identity")
    return dict(source_manifest_sha256=expected_manifest, source_pins=pins,
                source_checkpoint=str(D95), source_checkpoint_sha256=D95_SHA)



BASE_QUALIFIER_SHA = "b13fb5715d19410ce2f69f26f35853201eb076894d64856ea94cf3667790f67d"


def _numerical_parts(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    start = next(i for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "device" for t in node.targets))
    end = next(i for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "after" for t in node.targets))
    nodes = function.body[start:end]
    lines = source.splitlines()
    text = "\n".join(lines[nodes[0].lineno-1:nodes[-1].end_lineno]) + "\n"
    constants = {node.targets[0].id: ast.dump(node.value, include_attributes=False)
        for node in tree.body if isinstance(node, ast.Assign) and len(node.targets)==1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PRECISION", "CHECKS", "D95", "D95_SHA", "ARCHIVE")}
    precision_nodes = [node for node in function.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr in ("allow_tf32", "benchmark") for t in node.targets)]
    setup_start = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Expr)
        and ast.unparse(node).startswith("psutil.Process().nice("))
    setup_end = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "started" for t in node.targets))
    resource_nodes = [node for node in function.body if
        (isinstance(node, ast.If) and any(term in ast.unparse(node.test) for term in
            ("psutil.virtual_memory().available", "torch.cuda.is_available()", "torch.cuda.mem_get_info()")))
        or (isinstance(node, ast.Expr) and "torch.cuda.set_per_process_memory_fraction(" in ast.unparse(node))]
    return dict(text=text, AST=ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False),
        constants=constants, precision_AST=[ast.dump(node, include_attributes=False) for node in precision_nodes],
        setup_AST=[ast.dump(node, include_attributes=False) for node in function.body[setup_start:setup_end]],
        resource_AST=[ast.dump(node, include_attributes=False) for node in resource_nodes],
        original_line_range=[nodes[0].lineno, nodes[-1].end_lineno])


def verify_numerical_identity():
    reference = ROOT / "scripts/qualify_b1_full_actor_BC_CUDA.py"
    if sha(reference) != BASE_QUALIFIER_SHA:
        raise ValueError("The immutable b13 numerical reference qualifier differs")
    old = _numerical_parts(reference.read_text(encoding="utf-8"))
    new = _numerical_parts(Path(__file__).read_text(encoding="utf-8"))
    if any(old[key] != new[key] for key in ("text", "AST", "constants", "precision_AST", "setup_AST", "resource_AST")):
        raise ValueError("Numerical checks, inputs, loss, precision or constants differ from b13")
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    return dict(reference_script_sha256=BASE_QUALIFIER_SHA,
        numerical_fragment_LF_sha256=digest(old["text"]), numerical_fragment_AST_sha256=digest(old["AST"]),
        original_line_range=old["original_line_range"], lifecycle_line_range=new["original_line_range"],
        raw_fragment_and_AST_equal=True, constants_and_precision_AST_equal=True,
        original_CPU_setup_and_CUDA_limits_AST_equal=True,
        environment_guard_only_changed=True)


def load_lifecycle_config(args):
    from train_b1_actor_path_BC_lifecycle_v2 import validate_config
    outer = validate_config(read(args.lifecycle_config))
    bc, contract = outer["BC"], outer["main_contract"]
    progress = Path(contract["output"]).resolve() / "trial_000_both/progress.json"
    if (Path(bc["source_runtime"]).resolve() != ROOT.resolve()
            or bc["source_manifest_sha256"] != args.manifest_sha256
            or Path(bc["source_checkpoint"]).resolve() != D95.resolve()
            or bc["source_checkpoint_sha256"] != D95_SHA
            or Path(bc["demonstrations"]).resolve() != ARCHIVE.resolve()):
        raise ValueError("Lifecycle config must bind this final runtime and the same D95/raw30 inputs")
    for name, expected in (("main_pid", contract["pid"]), ("main_ctime", contract["create_time"])):
        value = getattr(args, name, None)
        if value is not None and value != expected:
            raise ValueError("Explicit main PID/ctime differs from lifecycle contract")
        setattr(args, name, expected)
    if args.main_progress is not None and args.main_progress.resolve() != progress:
        raise ValueError("Explicit main progress differs from lifecycle contract")
    args.main_progress = progress
    return outer


class QualificationEnvironment:
    def __init__(self, gate, outer, args, report, reader):
        self.gate, self.outer, self.args, self.report, self.reader = gate, outer, args, report, reader
        self.normal_completion_used = False

    def before_gpu(self):
        self.gate.before_gpu()
        evidence = deepcopy(self.gate.last_completion)
        self.normal_completion_used |= evidence is not None
        self.report.update(lifecycle_main_gate_contract=deepcopy(self.outer["main_contract"]),
            environment_guard_schema=self.outer["lifecycle"]["schema"],
            verified_normal_main_completion=evidence, main_environment_permission_checked=True,
            qualification_main_wait_seconds=self.gate.wait_seconds,
            qualification_main_wait_checks=self.gate.wait_checks,
            gate_is_not_a_cross_process_GPU_mutex=True)
        return self.reader(self.gate.progress_path)


def make_environment_gate(outer, args, report, started, *, guard_class=None, psutil_module=None,
                          clock=None, reader=None):
    from fly_touhou.thrl_actor_bc_lifecycle import LifecycleGuard
    if psutil_module is None:
        import psutil as psutil_module
    guard_class = guard_class or LifecycleGuard
    clock = clock or time.monotonic
    reader = reader or read
    def resource_check(*, startup, elapsed, stop_exists):
        minimum = (10 if startup else 6) * 1024**3
        if stop_exists or clock()-started >= 90 or psutil_module.virtual_memory().available < minimum:
            raise RuntimeError("Finite90second lifecycle qualification wall/RAM/STOP limit reached")
    gate = guard_class(outer["BC"], outer["main_contract"], args.output.parent, resource_check)
    gate.check(startup=True)
    def waiting(row, *, force=False):
        report["lifecycle_waiting_for_main"] = deepcopy(row)
    gate.on_wait = waiting
    return QualificationEnvironment(gate, outer, args, report, reader)


def qualify(args, report):
    import psutil
    import torch
    from train_b1_actor_path_BC_contrast import DATA_SHA
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    from fly_touhou.thrl_full_actor_path_bc import (
        ENCODER_PREFIXES, GRU_ACTOR_PREFIXES, FROZEN_PREFIXES,
        begin_actor_path_bc, forward_actor_path, hard_action_ce)
    from fly_touhou.thrl_parameter_transfer import read_evaluation_source
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits

    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()

    outer = load_lifecycle_config(args)
    environment_gate = make_environment_gate(outer, args, report, started, psutil_module=psutil)

    def main_status():
        return environment_gate.before_gpu()

    def guard():
        return environment_gate.before_gpu()

    if psutil.virtual_memory().available < 10 * 1024**3:
        raise RuntimeError("Need ten GiB available RAM before this independent numeric probe")
    before = guard()
    report["main_before"] = {key: before.get(key) for key in ("updated_utc", "actual_decisions", "adam_minibatch_steps", "stage")}
    report["CUDA_initialization_attempted"] = True
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA is required;no CPU acceptance fallback")
    torch.cuda.set_per_process_memory_fraction(.25)
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        raise RuntimeError("Need six GiB free device memory before a small probe")
    device = torch.device("cuda:0")
    source = read_evaluation_source(torch, D95, D95_SHA)
    dataset = FullTeacherDataset(ARCHIVE, reader_cache_bytes=20 * 1024**2)
    if dataset.manifest_sha256 != DATA_SHA or dataset.rows != 741839 or len(dataset.records) != 30:
        raise ValueError("Qualification requires the same sealed raw30 data as the contrast")
    report["demonstrations_manifest_sha256"] = dataset.manifest_sha256
    index = next(i for i, item in enumerate(dataset.records) if item["reader"].folder.name == "world000")
    reader = dataset.records[index]["reader"]
    independent_rows = [0, 10125, 22481]
    samples = [reader.fetch(row, row + 1) for row in independent_rows]

    def modalities(data, batch_major=True):
        guard()
        values = [torch.as_tensor(data[key], device=device) for key in ("features", "maps", "objects")]
        return [value[:, None] if batch_major else value[None] for value in values]

    import numpy as np
    probe = {key: np.concatenate([sample[key] for sample in samples]) for key in ("features", "maps", "objects", "actions", "dones")}
    f, m, o = modalities(probe)
    dones = torch.zeros(len(samples), 1, dtype=torch.bool, device=device)
    target = torch.as_tensor(probe["actions"], device=device)
    guard()

    def fresh(scope):
        guard()
        model = model_from_config(**source["model_config"]).to(device)
        report["actual_CUDA"] = True
        model.load_state_dict(source["model"], strict=True)
        metadata = begin_actor_path_bc(model, scope)
        model.gru.flatten_parameters()
        return model, metadata

    model, full_scope = fresh("full_actor_path")
    model.eval()
    hidden = torch.zeros(1, len(samples), model.gru_hidden_size, device=device)
    with torch.no_grad():
        original, _ = model.forward_sequence(f, m, hidden, dones, objects=o)
        actual, _ = forward_actor_path(model, f, m, o, hidden, dones)
        support_equal = torch.equal(torch.isfinite(original), torch.isfinite(actual))
        probability_error = float((policy_logits(original).softmax(-1) - policy_logits(actual).softmax(-1)).abs().max())
    if not support_equal or probability_error > 2e-5:
        raise ValueError("Actual CUDA final-wrapper parity failed")
    report["checks"]["final_wrapper_T1_parity"] = True
    report["final_wrapper_max_probability_error"] = probability_error

    prefix = reader.fetch(0, 4)
    pf, pm, po = modalities(prefix, batch_major=False)
    pd = torch.as_tensor(prefix["dones"], device=device)[None]
    h0 = torch.zeros(1, 1, model.gru_hidden_size, device=device)
    guard()
    with torch.no_grad():
        whole, following = forward_actor_path(model, pf, pm, po, h0, pd)
        first, h1 = forward_actor_path(model, pf[:, :2], pm[:, :2], po[:, :2], h0, pd[:, :2])
        last, h2 = forward_actor_path(model, pf[:, 2:], pm[:, 2:], po[:, 2:], h1, pd[:, 2:])
        block_error = float((whole.softmax(-1) - torch.cat((first, last)).softmax(-1)).abs().max())
        hidden_error = float((following - h2).abs().max())
    if block_error > 2e-5 or hidden_error > 2e-5:
        raise ValueError("Actual own-state prefix continuation failed")
    report["checks"]["own_hidden_block_continuation"] = True
    report["prefix_probability_error"] = block_error
    report["prefix_hidden_error"] = hidden_error
    guard()

    group_results = {}
    for scope in ("full_actor_path", "gru_actor_control"):
        if scope == "full_actor_path":
            del model
        model, metadata = fresh(scope)
        before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        logits, _ = forward_actor_path(model, f, m, o, hidden, dones)
        loss, statistics = hard_action_ce(logits, target)
        if not bool(torch.isfinite(loss)) or statistics["selected_labels"] < 1:
            raise ValueError("No finite legal real-state numeric CE")
        loss.backward()
        gradients = {}
        for prefix_name in (*ENCODER_PREFIXES, "base.map_encoder.spatial_attention."):
            selected = [p for name, p in model.named_parameters() if name.startswith(prefix_name)]
            gradients[prefix_name] = dict(tensors=len(selected), nonzero=sum(p.grad is not None and bool((p.grad != 0).any()) for p in selected),
                finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in selected))
        if any(not value["finite"] for value in gradients.values()):
            raise ValueError("Nonfinite actual encoder gradient")
        active = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in active):
            raise ValueError("Nonfinite actual actor-path gradient")
        norm = torch.nn.utils.clip_grad_norm_(active, .5)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite actual actor-path gradient norm")
        optimizer.step()
        report["numerical_probe_Adam_steps"] += 1
        after = model.state_dict()
        changed = {prefix_name: sum(not torch.equal(before[name], value.detach().cpu()) for name, value in after.items() if name.startswith(prefix_name))
                   for prefix_name in (*ENCODER_PREFIXES, "base.map_encoder.spatial_attention.")}
        frozen_equal = all(torch.equal(before[name], value.detach().cpu()) for name, value in after.items() if name.startswith(FROZEN_PREFIXES))
        actor_changes = {prefix_name: sum(not torch.equal(before[name], value.detach().cpu())
            for name, value in after.items() if name.startswith(prefix_name)) for prefix_name in GRU_ACTOR_PREFIXES}
        if not frozen_equal:
            raise ValueError("BC numeric step modified a critic or risk tensor")
        if any(value == 0 for value in actor_changes.values()):
            raise ValueError("An actual GRU/actor group did not update")
        if scope == "full_actor_path":
            if any(value["nonzero"] == 0 for value in gradients.values()) or any(value == 0 for value in changed.values()):
                raise ValueError("An actual visual/attention group had no gradient or change")
            report["checks"]["full_encoder_gradients_and_changes"] = True
        else:
            if any(changed.values()) or any(value["nonzero"] for value in gradients.values()):
                raise ValueError("Control unexpectedly updated an encoder")
            report["checks"]["control_encoder_unchanged"] = True
        group_results[scope] = dict(trainable_parameters=metadata["trainable_parameters"], loss=float(loss.detach()),
            statistics=statistics, encoder_gradients=gradients, changed_encoder_tensors=changed,
            changed_GRU_actor_tensors=actor_changes, pre_clip_gradient_norm=float(norm),
            all_actor_path_gradients_finite=True, critic_and_risk_unchanged=frozen_equal)
        optimizer.zero_grad(set_to_none=True)
        guard()
    report["checks"]["critic_and_risk_unchanged"] = True

    guard()
    no_stock = pf[:, :1].clone()
    no_stock[:, :, 4] = 0
    masked, _ = forward_actor_path(model, no_stock, pm[:, :1], po[:, :1], h0, pd[:, :1])
    excluded, stat = hard_action_ce(masked, torch.tensor([18], device=device), torch.tensor([False], device=device))
    if bool(torch.isfinite(masked[0, 18])) or not bool(torch.isfinite(excluded)) or float(excluded) != 0 or stat["selected_labels"] != 0:
        raise ValueError("Missing-inventory CE did not remain finite and excluded")
    report["checks"]["inventory_legal_CE_finite"] = True

    # Qualify the intended B8,L16 shape as well as the small numerical probes.
    # These eight real prefixes all start at source0; no optimizer is stepped.
    from fly_touhou.thrl_full_actor_bc_stream import SequentialRawTeacherBatches
    split = dataset.split(fraction=.2, seed=31)
    stream = SequentialRawTeacherBatches(dataset, sequence_length=16, batch_episodes=8, check_stop=guard)
    iterator = stream.epoch(split["train"], role="inspect")
    try:
        raw_batch = next(iterator)
    finally:
        iterator.close()
    if raw_batch.actions.shape != (8, 16) or not bool(raw_batch.episode_starts[:, 0].all()):
        raise ValueError("The planned batch is not eight actual source0 prefixes")
    planned, planned_scope = fresh("full_actor_path")
    inputs = [torch.as_tensor(getattr(raw_batch, key), device=device) for key in ("features", "maps", "objects")]
    own0 = torch.zeros(1, 8, planned.gru_hidden_size, device=device)
    planned_dones = torch.as_tensor(raw_batch.dones, device=device)
    torch.cuda.synchronize()
    batch_started = time.monotonic()
    batch_logits, _ = forward_actor_path(planned, *inputs, own0, planned_dones)
    batch_loss, batch_stats = hard_action_ce(batch_logits,
        torch.as_tensor(raw_batch.actions, device=device), torch.as_tensor(raw_batch.legal, device=device))
    if not bool(torch.isfinite(batch_loss)) or batch_stats["selected_labels"] < 1:
        raise ValueError("Nonfinite planned-size CUDA loss")
    batch_loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in planned.parameters()):
        raise ValueError("Nonfinite planned-size CUDA gradients")
    torch.cuda.synchronize()
    report["planned_batch"] = dict(episodes=8, sequence_length=16, rows=128,
        forward_backward_seconds=time.monotonic() - batch_started, legal_labels=batch_stats["selected_labels"],
        actual_source0_histories=True, optimizer_steps=0, not_a_steady_state_throughput_benchmark=True)
    report["checks"]["planned_batch_forward_backward"] = True
    guard()
    if sha(D95) != D95_SHA:
        raise ValueError("Immutable source checkpoint changed")
    dataset.close_caches()
    torch.cuda.synchronize()
    after = main_status()
    report.update(actual_CUDA=True, device=torch.cuda.get_device_name(), precision=PRECISION, scopes=group_results,
        independent_numeric_source_rows=independent_rows, late_zero_hidden_are_training_examples=False,
        real_continuous_prefix_source_rows=[0, 3], source_native_seed=dataset.records[index]["native_seed"],
        GPU_memory_fraction=.25, peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(), peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
        RAM_available_bytes=psutil.virtual_memory().available, elapsed_seconds=time.monotonic() - started,
        main_after={key: after.get(key) for key in ("updated_utc", "actual_decisions", "adam_minibatch_steps", "stage")},
        original_source_checkpoint_unchanged=True, child_processes_created=len(psutil.Process().children()),
        shared_GPU_timing_may_affect_main_ready_order=True,
        fresh_collecting_checked_before_each_GPU_block=not environment_gate.normal_completion_used,
        collecting_or_verified_normal_exit_checked_before_each_GPU_block=True,
        main_GPU_phase_is_not_a_mutual_exclusion_lock=True)
    if report["child_processes_created"] or not all(report["checks"].values()):
        raise RuntimeError("Unexpected child process or incomplete CUDA qualification")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--lifecycle-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-cuda", action="store_true")
    parser.add_argument("--main-pid", type=int)
    parser.add_argument("--main-ctime", type=float)
    parser.add_argument("--main-progress", type=Path)
    return parser.parse_args(argv)


def run(args):
    if args.output.exists():
        raise ValueError("Preserve existing qualification evidence;use a fresh output")
    report = dict(schema="B1-raw-actor-path-BC-actual-CUDA-qualification-v1", utc=utc_now(), status="not_started",
        actual_CUDA=False, precision=PRECISION, checks={key: False for key in CHECKS},
        CUDA_initialization_attempted=False, numerical_probe_Adam_steps=0, production_training_steps=0,
        native_worlds_started=0, deployment_checkpoint_saved=False, main_training_configuration_modified=False,
        previous_BC_attempt_resumed=False, main_environment_permission_checked=False)
    code = 0
    try:
        report["numerical_source_identity"] = verify_numerical_identity()
        outer = load_lifecycle_config(args)
        report.update(preflight(args.manifest_sha256))
        info = read(ROOT / "source_manifest.json")
        entries = {row["path"]: row["sha256"] for row in info["files"]}
        if ("src/fly_touhou/thrl_actor_bc_lifecycle.py" not in report["source_pins"]
                or entries.get("scripts/qualify_b1_full_actor_BC_lifecycle_CUDA.py") != sha(__file__)
                or entries.get("scripts/qualify_b1_full_actor_BC_CUDA.py") != BASE_QUALIFIER_SHA):
            raise ValueError("The lifecycle gate/new qualifier/reference must be pinned in the final runtime")
        report.update(lifecycle_config=dict(path=str(args.lifecycle_config.resolve()), sha256=sha(args.lifecycle_config)),
            lifecycle_main_gate_contract=deepcopy(outer["main_contract"]),
            environment_guard_schema=outer["lifecycle"]["schema"],
            qualification_limits=dict(wall_seconds=90, startup_RAM_GiB=10, running_RAM_GiB=6,
                startup_free_VRAM_GiB=6, GPU_memory_fraction=.25, torch_threads=2),
            normal_exit_permission_requires_full_original_completion_and_cleanup=True)
        if args.execute_cuda:
            qualify(args, report)
            report["status"] = "passed"
        else:
            report["status"] = "CPU_preflight_only_not_CUDA_acceptance"
    except Exception as error:
        code = 2
        report.update(status="failed", error=type(error).__name__ + ": " + str(error), no_automatic_retry=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({key: report.get(key) for key in
        ("status", "actual_CUDA", "checks", "numerical_probe_Adam_steps", "error", "elapsed_seconds")}))
    return code


def main():
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
