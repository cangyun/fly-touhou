"""Fit a TH08 policy's final route layer on measured original trajectories."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_rehearsals(args):
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal

    reference = SuccessRehearsal(args.reference_manifest, match_reference_scores=True)
    correction = SuccessRehearsal(args.correction_manifest[0],
                                  additional_manifests=args.correction_manifest[1:])
    for prefix, archive in (("retain", reference), ("correct", correction)):
        for index, row in enumerate(archive.rows):
            if not row.get("verified_native_exposure_rows") or row["native_validation"]["contacts"]:
                raise ValueError("Use continuous, contact-free replay from the original game")
            row["task_key"] = f"{prefix}-{index}"
    return reference, correction


def prepare_branch_openings(model, paths):
    import numpy as np
    import torch
    from fly_touhou.nbnm_prefix import burn_in
    from fly_touhou.nbnm_run import read

    openings = []
    with torch.no_grad():
        for path in paths:
            row = read(path)
            features = torch.as_tensor([row["features"]], dtype=torch.float32, device=model.device)
            with np.load(path.with_name("burn-in.npz"), allow_pickle=False) as source:
                history = {key: source[key] for key in ("features", "actions")}
            state = burn_in(model, {"policy_burn_in": history}, model.initial_state(1))
            output, _, shared = model.step(features, state, return_all_DN=True)
            descending = (output["candidate_DN"] if model.shared_directions is None
                          else model.shared_directions.directions(shared))
            clearance = torch.as_tensor(row["proposal_margins"][3:], device=model.device).transpose(0, 1)
            covered = torch.as_tensor(row["proposal_covered"][3:], device=model.device).transpose(0, 1)
            contact = torch.as_tensor([frame if frame is not None else 97
                                       for frame in row["proposal_first_contact"]], device=model.device)
            hit = contact[:, None] <= contact.new_tensor([48, 96])
            # A measured narrow exit still provides a valid pair against a collision.
            safe = covered & ~hit & (clearance > 0.)
            if not hit[:, 1].any() or not safe[:, 1].any():
                raise ValueError(f"Use a native opening containing both contact and viable routes: {path}")
            openings.append((features, descending, shared if model.shared_directions is not None else None,
                             clearance, hit, safe))
    return openings


def branch_opening_loss(model, opening):
    import torch
    from torch.nn import functional as F

    features, descending, pooled, clearance, hit, safe = opening
    output = model.heads(descending, features, pooled_DN=pooled)
    risk = output["corridor_logits"][0]
    known = hit | safe
    target = torch.where(hit, 1., 8. / (8. + clearance.clamp_min(0.)))
    risk_loss = F.binary_cross_entropy_with_logits(risk[known], target[known])
    witness = output["corridor_witness_logits"][0]
    witness_loss = F.binary_cross_entropy_with_logits(witness[known], safe[known].float())
    scores = output["logits"][0, :18]
    safe_scores = scores[safe[:, 1]]
    danger_scores = scores[hit[:, 1]]
    preference = F.softplus(danger_scores.max() - safe_scores.max() + .25)
    top = int(scores.argmax())
    return risk_loss + .5 * witness_loss + preference, dict(
        contact_candidates=int(hit[:, 1].sum()), viable_candidates=int(safe[:, 1].sum()),
        top_action=top, top_contact=bool(hit[top, 1]), risk_loss=float(risk_loss.detach()),
        safe_logit_lead=float((safe_scores.max() - danger_scores.max()).detach()))


def run(args, reference, correction, plan):
    import psutil
    import torch
    from fly_touhou.nbnm_run import atomic_torch_save, load_model, read, runtime
    from fly_touhou.th08_training import portable_spec
    from fly_touhou.thrl_durable_records import append_jsonl, write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    args.output.mkdir(parents=True)
    proc = psutil.Process()
    started = time.monotonic()
    temporary = runtime(args.output)

    def progress(phase, updates):
        write_json(args.output / "progress.json", dict(
            phase=phase, PID=proc.pid, create_time=proc.create_time(), completed_updates=updates,
            budget=args.updates, elapsed_seconds=time.monotonic()-started, online_decisions=0))

    try:
        write_json(args.output / "plan.json", plan)
        progress("loading", 0)
        spec = portable_spec(read(args.spec))
        model, migrated = load_model(spec, args.checkpoint, apply_native_sensors=False)
        if migrated or not (model.bombs_disabled and model.soft_risk_composition):
            raise ValueError("Use existing original TH08 NBNM parameters")
        model.eval().requires_grad_(False)
        for parameter in model.route[-1].parameters():
            parameter.requires_grad_(True)
        if args.branch_opening:
            model.corridor_head.requires_grad_(True)
            model.corridor_witness_head.requires_grad_(True)
        openings = prepare_branch_openings(model, args.branch_opening)
        trained = {name: param for name, param in model.named_parameters() if param.requires_grad}
        optimizer = torch.optim.Adam(trained.values(), lr=args.learning_rate)
        activity = {}
        for index in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            retained, retain_metrics = next(reference.iter_losses(
                model, index=index, task_key=reference.rows[index % len(reference.rows)]["task_key"],
                movement_activity=activity))
            (args.reference_weight * retained).backward()
            del retained
            corrected, correct_metrics = next(correction.iter_losses(
                model, index=index, task_key=correction.rows[index % len(correction.rows)]["task_key"],
                movement_activity=activity))
            corrected.backward()
            del corrected
            measured = None
            if openings:
                branch_loss, measured = branch_opening_loss(model, openings[index % len(openings)])
                branch_loss.backward()
                del branch_loss
            torch.nn.utils.clip_grad_norm_(trained.values(), max_norm=1., error_if_nonfinite=True)
            optimizer.step()
            step = index + 1
            append_jsonl(args.output / "updates.jsonl", dict(
                update=step, retention=retain_metrics, correction=correct_metrics, native_branch=measured))
            if step == args.updates // 2:
                atomic_torch_save(model.parameter_payload(dict(TH08_measured_route_updates=step)),
                                  args.output / f"parameters-{step:03d}.pt")
            progress("fitting", step)

        checkpoint = args.output / "latest-parameters.pt"
        atomic_torch_save(model.parameter_payload(dict(TH08_measured_route_updates=args.updates)), checkpoint)
        write_json(args.output / "result.json", dict(
            completed=True, checkpoint=str(checkpoint), parent=str(args.checkpoint),
            trainable_parameters=sorted(trained), optimizer_updates=args.updates, online_decisions=0,
            original_match_requires_frozen_evaluation=True))
        progress("complete", args.updates)
    except BaseException:
        write_json(args.output / "failure.json", dict(error=traceback.format_exc()))
        raise
    finally:
        cleanup_native_temp(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    path_type = lambda s: Path(s).resolve()
    for key in ("spec", "checkpoint", "reference-manifest", "output"):
        parser.add_argument(f"--{key}", type=path_type, required=True)
    parser.add_argument("--correction-manifest", type=path_type, action="append", required=True)
    parser.add_argument("--branch-opening", type=path_type, action="append", default=[],
                        help="Measured same-state candidate branches with their committed neural burn-in")
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--reference-weight", type=float, default=4.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.updates < 2 or not 0. < args.learning_rate < 1. or not 0. < args.reference_weight < float("inf"):
        parser.error("Specify a finite learning rate and retention weight with at least two updates")
    reference, correction = load_rehearsals(args)
    plan = dict(checkpoint=str(args.checkpoint), spec=str(args.spec),
                reference_manifest=str(args.reference_manifest),
                correction_manifests=[str(p) for p in args.correction_manifest],
                reference_histories=len(reference.rows), verified_corrections=len(correction.rows),
                train_scope="final route layer and corridor readouts" if args.branch_opening else "final route layer",
                measured_branch_openings=[str(p) for p in args.branch_opening], new_native_decisions=0,
                learning_rate=args.learning_rate, reference_weight=args.reference_weight, updates=args.updates)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    from fly_touhou.nbnm_job_phase import gpu_phase

    with gpu_phase("training"):
        run(args, reference, correction, plan)


if __name__ == "__main__":
    main()
