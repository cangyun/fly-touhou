"""Typed paid-final CPU inspect/export and explicit frozen development.

Default is CPU-only. An export plan cannot launch a game. Native plans keep
the new paid-final schema and delegate to the unchanged v1 runtime/exit/
capacity/private-desktop/greedy19/trace128 implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fly_touhou import thrl_actor_bc_final_evaluation as final_api
from fly_touhou import thrl_actor_bc_evaluation as selected_api
import evaluate_b1_actor_BC_candidate as selected_entry


def validate_plan(plan):
    if plan.get('schema') == final_api.EXPORT_PLAN_SCHEMA:
        if set(plan) != {'schema', 'final_checkpoint', 'selection'}:
            raise ValueError('A paid-final export plan requires only final.pt and actual selection.json references')
        return plan
    if plan.get('schema') != final_api.NATIVE_PLAN_SCHEMA:
        raise ValueError('An explicit paid-final v2 export or native plan is required')
    # Reuse only common plan-field validation. The candidate is never passed
    # to a v1 loader and retains its v2 schema throughout actual execution.
    normalized = selected_entry.validate_plan({**plan, 'schema': selected_api.NATIVE_PLAN_SCHEMA})
    return {**normalized, 'schema': final_api.NATIVE_PLAN_SCHEMA}


def verify_evaluation_context(plan, payload):
    final_api.verify_v1_bytes(ROOT)
    entries = selected_api.verify_manifest(ROOT, plan['evaluation_manifest_sha256'])
    required = ('src/fly_touhou/thrl_actor_bc_final_evaluation.py',
                'scripts/evaluate_b1_actor_BC_paid_final.py')
    if (Path(plan['evaluation_runtime']).resolve() != ROOT
            or any(name not in entries for name in required)
            or Path(final_api.__file__).resolve() != ROOT / required[0]
            or Path(__file__).resolve() != ROOT / required[1]):
        raise ValueError('The typed v2 module and entry must be included in this new frozen runtime')
    return selected_entry.verify_evaluation_context(plan, payload)


def run(args):
    plan = validate_plan(selected_api.read_json(args.plan))
    export_plan = plan['schema'] == final_api.EXPORT_PLAN_SCHEMA
    if (args.execute_native and export_plan) or (args.export_candidate and not export_plan):
        raise ValueError('Paid-final export and native plans are distinct')
    if (args.execute_native or args.export_candidate) and args.output is None:
        raise ValueError('Use an explicit fresh output directory')
    final_api.verify_v1_bytes(ROOT)
    if not args.execute_native:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import psutil
    import torch
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if export_plan:
        raw_final, provenance = final_api.load_paid_final(torch, plan['final_checkpoint'], plan['selection'])
        result = dict(schema='B1-paid-final-CPU-inspection-v2',
            artifact_role=final_api.ROLE, validation=provenance['paid_final_validation'],
            native_worlds_started=0, model_constructed=False, CUDA_initialized=torch.cuda.is_initialized())
        del raw_final
    else:
        payload = final_api.read_candidate(torch, plan['candidate'])
        cfg = verify_evaluation_context(plan, payload)
        result = final_api.candidate_summary(payload)
        result.update(model_constructed=False, CUDA_initialized=torch.cuda.is_initialized())
    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    args.output.mkdir(parents=True, exist_ok=False)
    args.created_output = args.output
    selected_entry.write_json(args.output / 'launch.json', dict(pid=os.getpid(),
        create_time=psutil.Process().create_time(), utc=selected_api.utc(),
        mode='paid-final-native-development' if args.execute_native else
        'paid-final-CPU-export' if args.export_candidate else 'paid-final-CPU-inspect',
        evaluation_source_schema=final_api.SCHEMA, artifact_role=final_api.ROLE,
        native_worlds_started=0, optimizer_updates=0, precision=selected_api.PRECISION))
    selected_entry.write_json(args.output / 'plan.json', plan)
    if args.export_candidate:
        artifact, summary = final_api.export_candidate(torch, plan['final_checkpoint'], plan['selection'],
            args.output / 'paid_final_for_development.pt')
        result.update(candidate=artifact, summary=summary)
    elif args.execute_native:
        report = final_api.execute_development(torch, plan, payload, cfg, args.output,
            delegate=selected_entry.execute_native)
        # Retain the unchanged backend's report as well as the typed v2
        # report;its candidate provenance already identifies paid-final.
        selected_entry.write_json(args.output / 'backend_result.json', selected_api.read_json(args.output / 'report.json'))
        selected_entry.write_json(args.output / 'report.json', report)
        return report
    selected_entry.write_json(args.output / 'inspection.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--export-candidate', action='store_true', help='CPU paid-final export only')
    modes.add_argument('--execute-native', action='store_true', help='Explicit typed paid-final development')
    return parser.parse_args(argv)


if __name__ == '__main__':
    arguments = parse_args()
    try:
        run(arguments)
    except BaseException:
        if getattr(arguments, 'created_output', None):
            selected_entry.write_json(arguments.created_output / 'failure.json', dict(
                utc=selected_api.utc(), traceback=traceback.format_exc(),
                evaluation_source_schema=final_api.SCHEMA, artifact_role=final_api.ROLE,
                optimizer_updates=0, partial_games_are_not_a_score=True, automatic_restart=False))
        raise
