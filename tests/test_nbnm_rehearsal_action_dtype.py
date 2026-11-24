import json

import numpy as np
import pytest
import torch

from fly_touhou.nbnm_model import configure_model
from fly_touhou.nbnm_rehearsal import SCHEMA, SuccessRehearsal
from test_compact_story_ready import build


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('action_dtype', [np.int32, np.int64])
def test_verified_correction_accepts_archive_integer_widths(tmp_path, device, action_dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    collector, model = build(tmp_path)
    configure_model(model, {'risk_mode': 'route'}, reset_critics=False)
    model.to(device)
    last = [layer for layer in model.risk.modules() if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(.125)
    features = np.pad(collector.obs['features'][0], (0, 152))[None].repeat(32, axis=0)
    np.savez(tmp_path / 'verified.npz', features=features,
        actions=np.ones(32, dtype=action_dtype), native_frames=np.arange(32) * 4,
        native_action_frames=np.full(32, 4), observed_contacts=np.zeros(32, bool))
    history = dict(file='verified.npz', task=dict(mode='stage', stage=1), split='train',
        decisions=32, nbnm_success=True, training_start=16,
        correction_windows=[[16, 32]], source_kind='native_verified_teacher_continuation',
        native_validation=dict(stage_clear=True, contacts=0, Bombs=0))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA, histories=[history])), encoding='utf-8')
    replay = SuccessRehearsal(manifest, native_short_risk_coefficient=.5,
        native_correction=dict(manifest=str(manifest), multiplier=2.))

    model.train()
    loss, info = replay.loss(model, index=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert info['rehearsal_native_correction']
    assert info['rehearsal_burn_in'] == 16
    assert info['rehearsal_risk_labels'] == 16
    assert info['rehearsal_native_false_alarm_rows'] == 16
    assert info['rehearsal_PPO_rows'] == 0
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.route.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.risk.parameters())
