"""Seal and CUDA-qualify a risk-policy prototype without starting native games."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a):
    import numpy as np
    import torch
    from fly_touhou.thrl_risk_policy import LearnedRiskPolicy, model_from_config
    from fly_touhou.thrl_training_mode import draw_native_seed
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA qualification is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = read(a.readout / 'result.json')
    parent_path = a.readout / 'base.pt'
    risk_path = a.readout / 'best.pt'
    parent_sha = digest(parent_path)
    if result['status'] != 'completed' or parent_sha != result['base_sha256']:
        raise ValueError('Risk fit or frozen parent is incomplete')
    if parent_sha != '4a5bab755f35ed5ba8f3581ffa312e5baf06b4eade6acd3282b38c4cb8d5dff1':
        raise ValueError('This experiment declares the retained parent, not the failed candidate')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    risk = torch.load(risk_path, map_location='cpu', weights_only=False)
    if risk['base_sha256'] != parent_sha or parent.get('optimizer_transaction_complete') is False:
        raise ValueError('Risk head belongs to a different or incomplete policy')
    model = LearnedRiskPolicy(parent['model_config'], input_dim=risk['input_dim'],
                              hidden_dim=risk['hidden_dim'], beta=0).cuda().eval()
    model.base.load_state_dict(parent['model'], strict=True)
    model.risk.load_state_dict(risk['model'], strict=True)
    # Reconstruct real native prefixes, preserving the actual original reset flags.
    folder = a.data / 'teacher30-v1' / 'demonstrations'
    arrays = {k: np.load(folder / (k + '.npy'), mmap_mode='r')
              for k in ['features', 'objects', 'maps', 'dones']}
    hidden = torch.zeros(1, 2, model.gru_hidden_size, device='cuda')
    with torch.inference_mode():
        for begin in range(0, 128, 16):
            cpu = {k: np.asarray(arrays[k][begin:begin+16, :2]).copy()
                   for k in ['features', 'objects', 'maps']}
            f = torch.as_tensor(cpu['features'].reshape(-1, 25), device='cuda')
            m = torch.as_tensor(cpu['maps'].reshape(-1, 16, 92, 96), device='cuda')
            o = torch.as_tensor(cpu['objects'].reshape(-1, 96, 21), device='cuda')
            encoded = model.base._encode(f, m, o).reshape(16, 2, -1)
            for local in range(16):
                _, hidden = model.base.gru(encoded[local:local+1], hidden.contiguous())
                done = torch.as_tensor(np.asarray(arrays['dones'][begin+local, :2]).copy(), device='cuda')
                hidden[:, done] = 0
        inputs = {k: torch.as_tensor(np.asarray(arrays[k][128, :2]).copy(), device='cuda')
                  for k in ['features', 'objects', 'maps']}
        f, m, o = (inputs[k] for k in ['features', 'maps', 'objects'])
        base_logits, base_v, base_h = model.base.forward_step(f, m, hidden, objects=o)
        zero, zv, zh = model.forward_step(f, m, hidden, objects=o)
        for expected, actual in [(base_logits, zero), (base_v, zv), (base_h, zh)]:
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        model.beta = 4.
        changed, cv, ch = model.forward_step(f, m, hidden, objects=o)
        bomb_error = float((changed.softmax(-1)[:, 18] - base_logits.softmax(-1)[:, 18]).abs().max())
        torch.testing.assert_close(ch, base_h, atol=0, rtol=0)
        torch.testing.assert_close(cv, base_v, atol=0, rtol=0)
        if bomb_error > 1e-6 or not torch.isfinite(changed[:, :18]).all():
            raise ValueError('Movement adjustment changed the Bomb marginal or legal movement support')
        encoded = model.base._encode(f, m, o)
        z = model.risk(torch.cat((encoded, base_h[0]), -1))
        if not torch.isfinite(z).all():
            raise ValueError('Nonfinite risk readout')
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    configuration = model.model_config()
    reconstructed = model_from_config(**configuration)
    reconstructed.load_state_dict(state, strict=True)
    del reconstructed
    native_seeds = []
    for game in range(20):
        env_seed, _ = np.random.SeedSequence([a.seed, game]).generate_state(2)
        native_seeds.append(draw_native_seed(np.random.default_rng(int(env_seed)), 'development'))
    if len(set(native_seeds)) != 20 or set(native_seeds) & {13266, 4906, 60366, 9207, 44216, 15136}:
        raise ValueError('Choose a fresh declared development seed before any games')
    if any(s % 10 not in (6, 7) for s in native_seeds):
        raise ValueError('Development seed partition differs')
    # Freeze from the qualified runtime, verifying every inherited file before copying.
    for p in [a.runtime, a.output]:
        if p.exists():
            raise FileExistsError(p)
    manifest = read(a.parent_runtime / 'source_manifest.json')
    a.runtime.mkdir(parents=True)
    for row in manifest['files']:
        src = a.parent_runtime / row['path']
        if digest(src) != row['sha256']:
            raise ValueError('Qualified source changed: ' + row['path'])
        dst = a.runtime / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    overlays = ['src/fly_touhou/thrl_action_risk_readout.py', 'src/fly_touhou/thrl_risk_policy.py',
                'scripts/evaluate_b1_risk_policy.py', 'scripts/evaluate_b1_teacher_avoidance.py',
                'scripts/prepare_b1_risk_evaluation.py', 'scripts/train_b1_action_risk_readout.py']
    for name in overlays:
        dst = a.runtime / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
    files = []
    for p in sorted(a.runtime.rglob('*')):
        if not p.is_file():
            continue
        if p.suffix == '.py':
            compile(p.read_bytes(), str(p), 'exec')
        files.append(dict(path=str(p.relative_to(a.runtime)), bytes=p.stat().st_size, sha256=digest(p)))
    write(a.runtime / 'source_manifest.json', dict(schema='fly-touhou-frozen-source-v1', utc=utc_now(),
        parent=str(a.parent_runtime), overlays=overlays, files=files,
        scope='Frozen parent plus learned neural movement readout; original native execution unchanged'))
    for name in ['src/fly_touhou/thrl_native_env.py', 'src/fly_touhou/thrl_full_story_env.py',
                 'src/fly_touhou/responsive_controller.py', 'src/fly_touhou/isolated_desktop_process.py',
                 'native/th14_fake_d3d9.dll']:
        if digest(a.runtime/name) != digest(a.parent_runtime/name):
            raise ValueError('Native execution changed')
    a.output.mkdir(parents=True)
    shutil.copyfile(parent_path, a.output/'initial.pt')
    torch.save(dict(schema='B1-risk-policy-evaluation-only-v1', model=state, model_config=configuration,
        optimizer_transaction_complete=True, evaluation_only=True, optimizer_recovery_eligible=False,
        candidate_optimized_decisions=0, inherited_decisions=245760, created_utc=utc_now(),
        lineage=dict(base_checkpoint_sha256=parent_sha, risk_checkpoint_sha256=digest(risk_path),
                     risk_best_epoch=risk['epoch'], risk_Adam_steps=risk['Adam_steps'],
                     risk_total_training_Adam_steps=result['Adam_steps'], movement_beta=4.,
                     native_efficacy_demonstrated=False)), a.output/'latest.pt')
    cfg = read(a.config)
    cfg.update(source_runtime=str(a.runtime), model_config=configuration,
               source_manifest_sha256=digest(a.runtime/'source_manifest.json'),
               evaluation_only=True, diagnostic_checkpoint_semantics='parent plus learned action-risk readout',
               initial_baseline_semantics='Same retained parent, already245760 PPO decisions; zero added risk/PPO updates')
    write(a.output/'config.json', cfg)
    qualification = dict(status='qualified_for_native_comparison', utc=utc_now(), device=str(torch.cuda.get_device_name(0)),
        native_instances=0, base_checkpoint_sha256=parent_sha, risk_checkpoint_sha256=digest(risk_path),
        policy_checkpoint_sha256=digest(a.output/'latest.pt'), beta0_logits_values_hidden_exact=True,
        beta4_hidden_values_exact=True, Bomb_marginal_max_absolute_error=bomb_error,
        all18_movements_legal=True, strict_roundtrip_load=True, fresh_native_development_seeds=native_seeds,
        evaluation_seed=a.seed, source_files=len(files), native_execution_byte_identical=True,
        risk_fit_sources='train partition0..5 only; grouped holdout for early stopping',
        native_efficacy_demonstrated=False)
    write(a.output/'CUDA-risk-policy-qualification-v1.json', qualification)
    print(json.dumps(qualification))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for key in ['readout', 'data', 'parent-runtime', 'runtime', 'output', 'config']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--seed', type=int, default=371593)
    run(p.parse_args())
