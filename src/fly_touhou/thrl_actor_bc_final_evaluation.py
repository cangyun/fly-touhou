"""Typed paid-final diagnostics, independent of heldout-CE selection.

The unchanged v1 reader verifies the actual selected policy and provenance.
This wrapper then checks final.pt, including its earlier cost snapshot, and
exports its own schema. It never relabels a rejected final as CE-selected.
No Torch, model, native world or optimizer is created on import.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import uuid

from . import thrl_actor_bc_evaluation as selected_api

SCHEMA = 'B1-actor-path-BC-paid-final-development-diagnostic-v2'
EXPORT_PLAN_SCHEMA = 'B1-actor-path-BC-paid-final-development-export-plan-v2'
NATIVE_PLAN_SCHEMA = 'B1-actor-path-BC-paid-final-full-story-development-plan-v2'
RESULT_SCHEMA = 'B1-actor-path-BC-paid-final-full-story-development-result-v2'
ROLE = 'paid-final diagnostic;not necessarily heldout-CE selected'
V1_PINS = {
    'src/fly_touhou/thrl_actor_bc_evaluation.py': 'ba4c15cc7dcd39cbc2e10ad6c9ccc645d8cf948218dc9643c27b2d41f2a837b4',
    'scripts/evaluate_b1_actor_BC_candidate.py': '21a09f98eb9e075b1ae6931c594c14b6728f122a19e7b985f3593537d7e083b4',
}
EVALUATION_CONTRACT = {**deepcopy(selected_api.EVALUATION_CONTRACT),
    'artifact_role': ROLE, 'heldout_CE_selection_required_for_this_artifact': False,
    'native_result_is_an_independent_measurement_not_CE_selection': True}
FINAL_HELD_COST_KEYS = ('final_heldout_source_rows_started', 'final_heldout_source_rows_completed')


def verify_v1_bytes(runtime):
    runtime = Path(runtime).resolve()
    for name, expected in V1_PINS.items():
        if selected_api.file_sha(runtime / name) != expected:
            raise ValueError('The unchanged v1 evaluator dependency differs: ' + name)


def _validate_final_after_selected_check(torch, final, selected, selection):
    """Only called after v1 has validated the real selected BC and source."""
    if (set(final) != selected_api.BC_FIELDS or final.get('schema') != selected_api.BC_SCHEMA
            or final.get('phase') != 'final-one-raw-epoch' or final.get('heldout') is not None):
        raise ValueError('An actual pre-heldout final-one-raw-epoch checkpoint is required')
    different_at_save = {'created_utc', 'phase', 'model', 'BC_optimizer', 'cost', 'heldout',
                         'model_BC_optimizer_steps'}
    if any(final[k] != selected[k] for k in selected_api.BC_FIELDS - different_at_save):
        raise ValueError('Paid-final scope, configuration, source or no-resume flags differ from selected')
    raw_cost, selection_cost = final['cost'], selection['cost']
    expected_earlier_cost = deepcopy(selection_cost)
    for key in FINAL_HELD_COST_KEYS:
        if type(selection_cost.get(key)) is not int or selection_cost[key] != 150283 or key in raw_cost:
            raise ValueError('The actual final-save/final-heldout chronology differs')
        expected_earlier_cost.pop(key)
    expected_earlier_cost['selected_BC_optimizer_steps'] = None
    if (raw_cost != expected_earlier_cost
            or type(final['model_BC_optimizer_steps']) is not int
            or final['model_BC_optimizer_steps'] != selection_cost['paid_BC_optimizer_steps']):
        raise ValueError('Final model steps must equal paid steps;only recorded heldout/selection costs may differ')
    before, after = selected['model'], final['model']
    trainable = set(selected['scope']['trainable_parameter_names'])
    if set(before) != set(after):
        raise ValueError('Paid-final NN tensor keys differ')
    changed_from_CE_selected = []
    for name, old in before.items():
        value = after[name]
        if (not torch.is_tensor(value) or value.device.type != 'cpu' or value.dtype != old.dtype
                or value.shape != old.shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Paid-final NN shape/dtype/finiteness differs: ' + name)
        if not torch.equal(value, old):
            if name not in trainable:
                raise ValueError('Paid-final changed a frozen encoder,critic,risk or persistent buffer: ' + name)
            changed_from_CE_selected.append(name)
    CE_selected = selection['selected']
    if CE_selected == 'final' and changed_from_CE_selected:
        raise ValueError('The CE-selected final must have exactly these final tensors')
    return selected_api.plain(dict(artifact_role=ROLE, source_phase='final-one-raw-epoch',
        heldout_CE_selected=CE_selected, final_was_heldout_CE_selected=CE_selected == 'final',
        paid_BC_optimizer_steps=selection_cost['paid_BC_optimizer_steps'],
        CE_selected_BC_optimizer_steps=selection_cost['selected_BC_optimizer_steps'],
        model_BC_optimizer_steps=final['model_BC_optimizer_steps'],
        source_final_cost_at_save=raw_cost, source_cost_after_final_heldout_and_CE_selection=selection_cost,
        final_heldout_metrics=selection['final_heldout'], initial_heldout_metrics=selection['initial_heldout'],
        changed_trainable_tensors_from_CE_selected=changed_from_CE_selected,
        all_frozen_tensors_equal_D95=True,
        cost_chronology='final.pt precedes final heldout;selection.json records later completed heldout and CE choice'))


def validate_paid_final_payload(torch, final, selected, ancestor, selection):
    """Pure tensor/metadata validation for CPU numerical contracts."""
    selected_api.validate_bc_payload(torch, selected, ancestor, selection)
    return _validate_final_after_selected_check(torch, final, selected, selection)


def load_paid_final(torch, final_ref, selection_ref):
    final_ref, selection_ref = selected_api.check_reference(final_ref), selected_api.check_reference(selection_ref)
    selection = selected_api.read_json(selection_ref['path'])
    selected_ref = dict(path=selection['selected_checkpoint'], sha256=selection['selected_checkpoint_sha256'])
    # This is the actual selected.pt, with its actual CE decision unchanged.
    selected, selected_provenance = selected_api.load_selected_bc(torch, selected_ref, selection_ref)
    final_path, selected_path = Path(final_ref['path']), Path(selected_ref['path']).resolve()
    if (final_path.name != 'final.pt' or final_path.parent != selected_path.parent
            or Path(selection_ref['path']).parent != selected_path.parent):
        raise ValueError('Explicit final.pt, selected.pt and selection.json must belong to the same completed arm')
    if selection['selected'] == 'final':
        if (Path(selection['selected_from']).resolve() != final_path
                or selection['selected_from_sha256'] != final_ref['sha256']):
            raise ValueError('The CE-selected final reference differs')
    final = selected_api.restricted_load(torch, final_ref['path'])
    facts = _validate_final_after_selected_check(torch, final, selected, selection)
    provenance = deepcopy(selected_provenance)
    provenance['CE_selected_validation'] = provenance.pop('validation')
    provenance.update(source_BC=final_ref, source_final_BC=final_ref,
        source_CE_selected_BC=selected_api.check_reference(selected_ref),
        source_phase='final-one-raw-epoch', artifact_role=ROLE,
        paid_final_validation=facts,
        model_BC_optimizer_steps=facts['model_BC_optimizer_steps'],
        paid_BC_optimizer_steps=facts['paid_BC_optimizer_steps'],
        CE_selected_BC_optimizer_steps=facts['CE_selected_BC_optimizer_steps'],
        final_was_heldout_CE_selected=facts['final_was_heldout_CE_selected'],
        source_final_cost_at_save=facts['source_final_cost_at_save'],
        cost_semantics='cost is the real CE-selection ledger;model_BC_optimizer_steps independently describes this paid-final artifact')
    # provenance.cost and provenance.selection remain the actual CE selection
    # record. They are not rewritten to pretend that an unselected final won.
    return final, selected_api.plain(provenance)


def candidate_summary(payload):
    provenance = payload['provenance']
    facts = provenance['paid_final_validation']
    return selected_api.plain(dict(schema=SCHEMA, artifact_role=ROLE,
        source_final_BC=provenance['source_final_BC'], source_CE_selected_BC=provenance['source_CE_selected_BC'],
        heldout_CE_selected=provenance['selection']['selected'],
        final_was_heldout_CE_selected=facts['final_was_heldout_CE_selected'],
        paid_BC_optimizer_steps=facts['paid_BC_optimizer_steps'],
        CE_selected_BC_optimizer_steps=facts['CE_selected_BC_optimizer_steps'],
        model_BC_optimizer_steps=facts['model_BC_optimizer_steps'],
        scope=provenance['scope']['scope'], final_heldout_metrics=facts['final_heldout_metrics'],
        source_D95=provenance['source_D95'], source_final_cost_at_save=provenance['source_final_cost_at_save'],
        source_cost_after_final_heldout_and_CE_selection=provenance['cost'],
        evaluation_contract=payload['evaluation_contract'], artifact_contains_native_performance=False))


def export_candidate(torch, final_ref, selection_ref, destination):
    final, provenance = load_paid_final(torch, final_ref, selection_ref)
    payload = dict(schema=SCHEMA, created_utc=selected_api.utc(), model_config=deepcopy(final['model_config']),
        model={name: value.detach().cpu().clone() for name, value in final['model'].items()},
        provenance=provenance, evaluation_contract=deepcopy(EVALUATION_CONTRACT))
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Paid-final diagnostic exports are immutable;use a new destination')
    temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return selected_api.reference(destination, selected_api.file_sha(destination)), candidate_summary(payload)


def read_candidate(torch, candidate_ref):
    candidate_ref = selected_api.check_reference(candidate_ref)
    payload = selected_api.restricted_load(torch, candidate_ref['path'])
    if (set(payload) != selected_api.CANDIDATE_FIELDS or payload.get('schema') != SCHEMA
            or payload.get('evaluation_contract') != EVALUATION_CONTRACT):
        raise ValueError('Explicit paid-final diagnostic schema without optimizer,RNG or hidden is required')
    final, provenance = load_paid_final(torch, payload['provenance']['source_final_BC'],
                                       payload['provenance']['selection_record'])
    if (payload['provenance'] != provenance or payload['model_config'] != final['model_config']
            or set(payload['model']) != set(final['model'])
            or any(not torch.is_tensor(value) or value.device.type != 'cpu'
                or value.shape != final['model'][name].shape or value.dtype != final['model'][name].dtype
                or not torch.equal(value, final['model'][name]) for name, value in payload['model'].items())):
        raise ValueError('Paid-final diagnostic export differs from the actual completed arm')
    return payload


def execute_development(torch, plan, payload, config, output, *, delegate):
    """Thin typed delegation to unchanged v1 execute_native (or a CPU fake).

    Callers must first use read_candidate and the v1 runtime/capacity preflight.
    No candidate/selection schema is rewritten before calling the delegate.
    """
    if (plan.get('schema') != NATIVE_PLAN_SCHEMA or payload.get('schema') != SCHEMA
            or payload.get('evaluation_contract') != EVALUATION_CONTRACT):
        raise ValueError('The paid-final typed plan and artifact are required before native delegation')
    source = selected_api.check_reference(plan['candidate'])
    # Revalidation is bounded CPU reading;it never constructs a model. This
    # makes the public delegation function reject an unchecked payload too.
    checked = read_candidate(torch, source)
    if (checked['provenance'] != payload['provenance'] or checked['model_config'] != payload['model_config']
            or set(checked['model']) != set(payload['model'])
            or any(not torch.is_tensor(payload['model'][name])
                or payload['model'][name].dtype != value.dtype or payload['model'][name].shape != value.shape
                or not torch.equal(value, payload['model'][name]) for name, value in checked['model'].items())):
        raise ValueError('The delegated paid-final payload differs from its pinned artifact')
    report = delegate(torch, plan, payload, {**config, 'evaluation_only_policy_role': SCHEMA}, output)
    return {**report, 'backend_result_schema': report.get('schema'), 'schema': RESULT_SCHEMA,
        'artifact_role': ROLE, 'evaluation_source_schema': SCHEMA,
        'paid_final_diagnostic': candidate_summary(payload),
        'heldout_CE_choice_was_not_modified': True, 'deployment_forbidden': True}
