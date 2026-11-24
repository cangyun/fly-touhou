"""Train and time a single-state student against archived NBNM policy outputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from torch.nn import functional as F

from fly_touhou.compact_story_model import sequence
from fly_touhou.nbnm_cuda import InferenceGraphs
from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.nbnm_prefix import apply_resets, replay_resets
from fly_touhou.nbnm_run import atomic_torch_save, load_cpu_checkpoint, load_model, read
from fly_touhou.spiking_engineered_control import SpikingState
from fly_touhou.thrl_durable_records import append_jsonl, write_json


FIELD_MAP = {
    "features": "features", "episode_starts": "starts", "actions": "actions",
    "recorded_policy_logits": "teacher_logits", "recorded_neural_horizon_scores": "teacher_risk",
    "recorded_corridor_risk_logits": "teacher_corridor", "recorded_corridor_witness_logits": "teacher_witness",
    "recorded_corridor_course_logits": "teacher_course", "old_values": "teacher_values",
    "recorded_native_stage": "stage", "margins": "margins", "trusted": "trusted",
    "recorded_observed_contacts": "observed_contacts", "unprotected_mask": "unprotected",
    "native_action_frames": "frames",
}


def windows(tape, worlds):
    return [(t, w) for t, block in tape["block_states"].items() for w in block.worlds if w in worlds]


def load_windows(tape, group, device):
    first, world = zip(*group)
    ticks = torch.arange(16)[:, None] + torch.tensor(first)
    pack = {dest: tape[source][ticks, world].to(device) for source, dest in FIELD_MAP.items()}
    state = SpikingState.pack([tape["block_states"][t].row(w) for t, w in group]).to(device)
    resets = replay_resets(tape.get("episode_initial_states", {}), group, 16, device)
    return pack, state, resets


def imitate(model, pack, state, resets):
    out = sequence(model, pack["features"], pack["starts"], state, pack["actions"],
                   checkpoint_chunks=True, reset_states=resets)
    play = pack["features"][..., 14] > .5
    student = out["logits"][..., :18][play].log_softmax(dim=-1)
    teacher = pack["teacher_logits"][..., :18][play].log_softmax(dim=-1)
    policy = F.kl_div(student, teacher, log_target=True, reduction="batchmean")
    # Keep independently learned short-horizon and corridor signals, even
    # when the same action wins after composing the heads.
    risk = F.smooth_l1_loss(8 * out["horizon_scores"][play], 8 * pack["teacher_risk"][play])
    corridor = F.smooth_l1_loss(out["corridor_logits"][play], pack["teacher_corridor"][play])
    witness = F.smooth_l1_loss(out["corridor_witness_logits"][play], pack["teacher_witness"][play])
    course = F.smooth_l1_loss(out["corridor_course_logits"][play], pack["teacher_course"][play])
    critic = F.smooth_l1_loss(out["values"], pack["teacher_values"])
    contact = play & pack["unprotected"].bool() & (pack["frames"] == 4)
    selected = (8 * out["horizon_scores"][..., 0] - 1).gather(-1, pack["actions"].clamp_max(17)[..., None])[..., 0]
    factual = F.binary_cross_entropy_with_logits(selected, pack["observed_contacts"].float(), reduction="none")
    factual = (factual * contact).sum() / contact.sum().clamp_min(1)
    return policy + .25 * risk + .1 * (corridor + witness + course + critic + factual)


@torch.no_grad()
def validate(model, tape, heldout):
    """Replay complete held-out world histories, carrying the student's own state."""
    first = SpikingState.pack([tape["block_states"][0].row(w) for w in heldout]).to(model.device)
    state = first
    length = int(tape["valid_mask"][:, heldout].sum(0).max())
    fields = (*FIELD_MAP, "valid_mask")
    data = {key: tape[key][:length, heldout].to(model.device) for key in fields}
    resets = {}
    for (tick, world), reset in tape.get("episode_initial_states", {}).items():
        if world in heldout:
            resets.setdefault(tick, []).append((heldout.index(world), reset.to(model.device)))
    graph = InferenceGraphs(model, sample=False, packed_binary_messages=True)
    decisions, risks, kls = [], [], []
    for tick in range(length):
        state = apply_resets(state.reset(data["episode_starts"][tick]), resets.get(tick))
        out, state, _ = graph.frozen_batch(data["features"][tick], state, len(heldout))
        logits = out["logits"][..., :18]
        decisions.append(logits.argmax(-1))
        risks.append(out["horizon_scores"][..., 0])
        logp = logits.log_softmax(-1)
        teacher = data["recorded_policy_logits"][tick, :, :18].log_softmax(-1)
        term = torch.where(torch.isfinite(teacher), teacher.exp() * (teacher - logp), 0.)
        kls.append(term.sum(-1))
    graph.clear()
    chosen = torch.stack(decisions)
    scores = torch.stack(risks)
    old = data["recorded_policy_logits"][..., :18].argmax(-1)
    available = data["valid_mask"] & (data["features"][..., 14] > .5)
    known = available & data["trusted"][..., 0] & data["unprotected_mask"].bool()
    unsafe = data["margins"][..., 0, :] <= 0
    avoidable = known & unsafe.any(-1) & (~unsafe).any(-1)
    chooses_unsafe = unsafe.gather(-1, chosen[..., None])[..., 0]
    teacher_unsafe = unsafe.gather(-1, old[..., None])[..., 0]
    # Rank unsafe actions above safe actions in predicted risk, within each scene.
    pairs = avoidable[..., None, None] & unsafe[..., :, None] & (~unsafe)[..., None, :]
    predicted = scores[..., :, None] - scores[..., None, :]
    comparable = (predicted > 0).float() + .5 * (predicted == 0)
    kl = torch.stack(kls)
    result = {}
    for stage in range(1, 7):
        selected = available & (data["recorded_native_stage"] == stage)
        count = int(selected.sum())
        if not count:
            continue
        marked = selected & avoidable
        stage_pairs = pairs & selected[..., None, None]
        risk_pairs = int(stage_pairs.sum())
        result[str(stage)] = dict(control_decisions=count, greedy_matches=int(((chosen == old) & selected).sum()),
            policy_kl=float(kl[selected].mean()), route_choice_scenes=int(marked.sum()),
            chosen_in_predicted_danger=int((chooses_unsafe & marked).sum()),
            teacher_in_predicted_danger=int((teacher_unsafe & marked).sum()), risk_pairs=risk_pairs,
            risk_rank_accuracy=float(comparable[stage_pairs].mean()) if risk_pairs else None)
    return result


@torch.no_grad()
def inference_rate(model, x, batch=7):
    graph = InferenceGraphs(model, sample=False, packed_binary_messages=True)
    state = model.initial_state(batch)
    # Reuse a CUDA graph and measure per-world state copy as in collection.
    for item in x[:4]:
        _, state, _ = graph.frozen_batch(item[:batch], state, batch)
    torch.cuda.synchronize()
    start = perf_counter()
    for item in x:
        _, state, _ = graph.frozen_batch(item[:batch], state, batch)
    torch.cuda.synchronize()
    wall = perf_counter() - start
    graph.clear()
    return dict(seconds=wall, decisions=len(x)*batch, decisions_s=len(x)*batch/wall, batch=batch)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["FLY_TOUHOU_SPMM_CACHE"] = str(args.output / "spmm-kernel-cache")
    os.environ["FLY_TOUHOU_EDGE_GRAD_CACHE"] = str(args.output / "kernel-cache")
    with gpu_phase("training"):
        torch.manual_seed(43)
        torch.cuda.manual_seed_all(43)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(2)
        spec = read(args.configuration)
        tape = load_cpu_checkpoint(args.rollout)
        model, _ = load_model(spec, args.checkpoint)
        model.core.fused_lif_tick = True
        heldout = tuple(args.heldout_worlds)
        train = windows(tape, set(range(tape["valid_mask"].size(1))) - set(heldout))
        eval_group = windows(tape, heldout)
        rng = np.random.default_rng(61)
        # Weight local windows by inverse square-root stage frequency, and
        # preferentially see situations with safe and dangerous directions.
        stage = np.array([int(tape["recorded_native_stage"][t, w]) for t, w in train])
        freq = np.bincount(stage)
        threat = np.array([bool(((tape["margins"][t:t+16, w, 0] <= 0).any(-1)
             & (tape["margins"][t:t+16, w, 0] > 0).any(-1)).any()) for t, w in train])
        weights = (1 + 2*threat) / np.sqrt(freq[stage])
        weights /= weights.sum()
        examples = tape["features"][:80, :7].to(model.device)
        reference = {}
        for fused in (False, True):
            model.core.fused_lif_tick = fused
            reference["fused" if fused else "torch"] = {str(b): inference_rate(model, examples, b)
                for b in (1, 7)}
        write_json(args.output / "reference-inference.json", reference)
        print(json.dumps(reference), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        model.enable_shared_directions()
        model.eval()
        before = validate(model, tape, heldout)
        write_json(args.output / "before.json", before)
        print("Measured untrained single-state policy on held-out worlds", flush=True)
        groups = [
            dict(params=list(model.core.internal_parameters()), lr=3e-6),
            dict(params=list(model.shared_directions.parameters()), lr=6e-4),
        ]
        used = {id(p) for group in groups for p in group["params"]}
        groups.append(dict(params=[p for p in model.parameters() if id(p) not in used], lr=1e-4))
        opt = torch.optim.Adam(groups, fused=True)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(args.updates):
            picked = rng.choice(len(train), args.batch, replace=False, p=weights)
            group = [train[index] for index in picked]
            start = perf_counter()
            inputs = load_windows(tape, group, model.device)
            opt.zero_grad(set_to_none=True)
            loss = imitate(model, *inputs)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            norm, loss = float(norm), float(loss)
            if not np.isfinite(norm + loss):
                raise FloatingPointError("Nonfinite shared-direction pilot update")
            times.append(perf_counter()-start)
            if (i+1) % 20 == 0 or i+1 == args.updates:
                row = dict(update=i+1, loss=loss, gradient_norm=norm, seconds=sum(times))
                append_jsonl(args.output / "learning.jsonl", row)
                print(json.dumps(row), flush=True)
        memory = torch.cuda.max_memory_allocated()
        opt.zero_grad(set_to_none=True)
        del opt, loss, inputs
        model.eval()
        after = validate(model, tape, heldout)
        speed = {str(b): inference_rate(model, examples, b) for b in (1, 7)}
        report = dict(heldout_worlds=heldout, train_windows=len(train), heldout_windows=len(eval_group),
            timings=dict(reference=reference, single_state=speed), before=before, after=after,
            updates=args.updates, batch=args.batch, peak_allocated_bytes=memory, learning_seconds=sum(times),
            median_update_seconds=float(np.median(times[1:] or times)),
            danger_label="trusted geometric four-frame margins on archived observations",
            game_control_evaluated=False)
        write_json(args.output / "report.json", report)
        atomic_torch_save(model.parameter_payload(dict(distillation_updates=args.updates)),
                          args.output / "single-state-parameters.pt")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("configuration", "checkpoint", "rollout", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--heldout-worlds", type=int, nargs="+", default=[2, 4, 7, 9, 10, 14])
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    run(parser.parse_args())
