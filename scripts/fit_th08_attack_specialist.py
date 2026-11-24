"""Fit a TH08 attack readout from a fully recorded original, contact-free route."""

import argparse
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.nbnm_fixed_movement_activity import capture_movement_activity
from fly_touhou.nbnm_job_phase import gpu_phase
from fly_touhou.nbnm_run import atomic_torch_save, load_model, read, runtime
from fly_touhou.th08_attack_readout import ATTACK_AGE_COLUMN, CONTEXT_COUNT, OBSERVATION_COLUMN
from fly_touhou.th08_training import portable_spec
from fly_touhou.thrl_durable_records import append_jsonl, write_json
from fly_touhou.thrl_job_paths import cleanup_native_temp


def movement_logits(model, features, actions, descending, pooled):
    prediction = model.heads(descending, features, pooled_DN=pooled)
    model._choose(prediction, features, actions=actions, sample=False,
                  action_selector=None, previous_actions=None)
    return prediction["logits"][:, :18]


def rank_gap(logits, chosen):
    selected = logits.gather(1, chosen[:, None]).squeeze(1)
    other = logits.masked_fill(torch.nn.functional.one_hot(chosen, 18).bool(), -torch.inf)
    return selected - other.amax(1)


def guide_data(folder, device):
    certificate = read(folder / "report.json")
    if (not certificate.get("teacher_only") or not certificate.get("complete_attack")
            or not certificate.get("zero_contact")
            or any(certificate.get(key) != 0 for key in ("misses", "bombs", "continues"))):
        raise ValueError("Supply a complete attack executed in the original with zero contact and Bomb")
    with np.load(folder / "matched-history.npz", allow_pickle=False) as recorded:
        features = torch.as_tensor(recorded["features"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(recorded["actions"], dtype=torch.long, device=device)
    first = certificate["training_start"]
    if (features.shape != (len(actions), 407) or not torch.isfinite(features).all()
            or not ((actions >= 0) & (actions < 18)).all()
            or not 0 <= first < len(actions) or first + certificate["verified_guide_decisions"] != len(actions)):
        raise ValueError("The original action tape and observations must match the certified route")
    return certificate, features, actions, first


def run(args):
    args.output.mkdir(parents=True)
    started = time.monotonic()
    temporary = runtime(args.output)

    def progress(phase, updates, **detail):
        write_json(args.output / "progress.json", dict(
            phase=phase, updates=updates, elapsed_seconds=time.monotonic() - started, **detail))

    try:
        progress("loading", 0)
        spec = portable_spec(read(args.spec))
        model, migrated = load_model(spec, args.checkpoint, apply_native_sensors=False)
        if migrated or model.native_attack_readout is None or not model.bombs_disabled:
            raise ValueError("Choose the observed-attack TH08 NBNM connectome checkpoint")
        model.eval().requires_grad_(False)
        certificate, features, actions, first = guide_data(args.guide, model.device)
        context = certificate["spell"] + 1
        recorded_context = (features[:, OBSERVATION_COLUMN] * CONTEXT_COUNT).round().long()
        if not (recorded_context[first:] == context).all():
            raise ValueError("All supervised moves must belong to the certified native attack")
        opening_age = float(features[first, ATTACK_AGE_COLUMN])
        opening_frames = round(600 * opening_age / (1 - opening_age))
        first_active_frame = max(0, opening_frames - 1)
        minimum_age = first_active_frame / (first_active_frame + 600)
        readout = model.native_attack_readout
        key = str(context)
        specialist = readout.specialists[key] if key in readout.specialists else None
        if specialist is not None:
            minimum_age = float(specialist.minimum_age)
        if not (features[first:, ATTACK_AGE_COLUMN] >= minimum_age).all():
            raise ValueError("The certified route must stay in the observed attack's enabled period")
        if first and bool(((recorded_context[:first] == context)
                          & (features[:first, ATTACK_AGE_COLUMN] >= minimum_age)).any()):
            raise ValueError("The specialized readout would change decisions before the verified route")
        starts = torch.zeros((len(actions), 1), device=model.device, dtype=torch.bool)
        starts[0] = True
        progress("recorded-neural-activity", 0, burn_in=first, route_decisions=len(actions) - first)
        activity = capture_movement_activity(model, features[:, None], starts, actions[:, None], model.initial_state(1))
        descending = torch.cat([row[0] for row in activity])
        pooled = None if activity[0][1] is None else torch.cat([row[1] for row in activity])
        del activity
        with torch.no_grad():
            baseline = movement_logits(model, features, actions, descending, pooled)
        if specialist is None:
            specialist = readout.add_specialist(context, minimum_age)
        specialist.requires_grad_(True)
        optimizer = torch.optim.Adam(specialist.parameters(), lr=args.learning_rate)
        active = slice(first, None)
        tape = (features[active], actions[active], descending[active],
                None if pooled is None else pooled[active])
        target = actions[active]
        before = int((baseline[active].argmax(-1) == target).sum())
        best = None
        best_weights = None
        chosen_update = 0
        for update in range(1, args.steps + 1):
            if (args.output / "STOP").exists():
                write_json(args.output / "report.json", dict(operator_requested_stop=True, updates=update-1))
                progress("stopped", update-1)
                return
            optimizer.zero_grad(set_to_none=True)
            gap = rank_gap(movement_logits(model, *tape), target)
            slack = (.18 - gap).clamp_min(0)
            loss = slack.square().mean() + .2 * slack[:16].square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(specialist.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if update % 16 != 0 and update != args.steps:
                continue
            with torch.no_grad():
                logits = movement_logits(model, *tape)
                margin = rank_gap(logits, target)
                matched = int((logits.argmax(-1) == target).sum())
                opening = int((logits[:16].argmax(-1) == target[:16]).sum())
                quality = (matched, opening, float(margin.min()))
                if best is None or quality > best:
                    best = quality
                    chosen_update = update
                    best_weights = {name: param.detach().clone() for name, param in specialist.state_dict().items()}
            row = dict(matched=matched, route_decisions=len(target), opening=opening, minimum_rank_gap=float(margin.min()))
            append_jsonl(args.output / "updates.jsonl", dict(update=update, **row))
            progress("fitting-attack-readout", update, **row)
            if matched == len(target) and quality[-1] >= .08:
                break
        specialist.load_state_dict(best_weights)
        with torch.no_grad():
            replay = movement_logits(model, features, actions, descending, pooled)
            if not torch.equal(baseline[:first], replay[:first]):
                raise RuntimeError("The specialist changed an observed decision before activation")
            fitted = int((replay[active].argmax(-1) == target).sum())
        name = "best-parameters.pt" if fitted == len(target) else "trial-parameters.pt"
        selected = args.output / name
        payload = model.parameter_payload(dict(TH08_attack_context=context, TH08_fit_updates=chosen_update))
        if spec.get("run_contract"):
            from fly_touhou.training.run_contract import mark_parameters
            mark_parameters(payload, dict(spec["run_contract"], native_attack_context=context,
                                          minimum_attack_age=minimum_age))
        atomic_torch_save(payload, selected)
        result = dict(parent=str(args.checkpoint), guide=str(args.guide), certified_attack=certificate["spell"],
                      fit_start=certificate["native_start"], fit_end=certificate["native_end"],
                      activation_age=minimum_age, before_matched=before, matched=fitted,
                      route_decisions=len(target), prefix_scores_verified=first, fit_updates=update,
                      selected_update=chosen_update, checkpoint=str(selected), original_rollout_pending=True)
        write_json(args.output / "result.json", result)
        progress("complete", update, matched=fitted, route_decisions=len(target))
    except BaseException:
        write_json(args.output / "failure.json", dict(error=traceback.format_exc()))
        raise
    finally:
        cleanup_native_temp(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("spec", "checkpoint", "guide", "output"):
        parser.add_argument(f"--{field}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=.001)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.learning_rate < 1:
        parser.error("Choose a positive fitting budget and learning rate")
    with gpu_phase("training"):
        run(args)
