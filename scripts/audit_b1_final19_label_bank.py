"""Read-only, stratified audit of a completed final19 bank; starts no models."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_durable_records import write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(bank, numeric, output):
    if output.exists():
        raise FileExistsError('Preserve the existing audit')
    result, identity = read(bank / 'result.json'), read(bank / 'identity.json')
    identity_sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if (result.get('status') != 'completed' or result.get('committed_rows') != 741839
            or len(result.get('committed_episodes', [])) != 30
            or result.get('identity_sha256') != identity_sha
            or identity.get('save_final_logits') is not True
            or identity['context']['teacher_sha256'] !=
            'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'):
        raise ValueError('Require the completed immutable D95 final19 bank')
    episodes, critical, total, enlarged_total = [], [], 0, 0
    sampled_rows = sampled_shards = 0
    for receipt_path in sorted(numeric.glob('game*-receipt.json')):
        receipt = read(receipt_path)
        compact = Path(receipt['compact'])
        if digest(compact) != receipt['compact_sha256']:
            raise ValueError('Independent numeric archive changed')
        name = Path(receipt['source_index']).parent.name
        folder = bank / 'episodes' / name
        commit, metadata = read(folder / 'COMMITTED.json'), read(folder / 'episode_index.json')
        source = metadata['source_episode']
        files = {item['name']: item for item in commit['files']}
        if (commit['identity_sha256'] != identity_sha
                or source['index_sha256'] != receipt['source_index_sha256']
                or digest(Path(source['index'])) != receipt['source_index_sha256']
                or digest(folder / 'episode_index.json') != files['episode_index.json']['sha256']
                or source['native_seed'] != receipt['native_seed']):
            raise ValueError('Bank and independently verified source episode differ')
        with np.load(compact, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ('features', 'before_frames',
                'after_observation_frames', 'native_seeds', 'episode_starts', 'dones', 'stage')}
        f = arrays['features']
        n = len(f)
        if n != metadata['rows'] or n != commit['rows']:
            raise ValueError('Committed source episode length differs')
        usable = (f[:, 24] > .5) & (f[:, 4] > 0) & (f[:, 19] >= .25) & (f[:, 19] <= 1)
        giant = f[:, 13] * 64 > 3
        chosen = {0, n - 1, *np.flatnonzero(usable).tolist()}
        for mask in [arrays['stage'] == stage for stage in range(1, 7)] + [giant, f[:, 4] <= 0]:
            indices = np.flatnonzero(mask)
            if len(indices):
                chosen.update(indices[np.linspace(0, len(indices) - 1, min(3, len(indices)), dtype=int)].tolist())
        cache = {}
        for row in sorted(chosen):
            number, offset = divmod(row, 32)
            spec = metadata['label_shards'][number]
            if number not in cache:
                path = folder / spec['name']
                if (path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']
                        or files[spec['name']]['sha256'] != spec['sha256']):
                    raise ValueError('Selected committed label shard changed')
                with np.load(path, allow_pickle=False) as archive:
                    cache[number] = {key: archive[key] for key in ('teacher_probabilities',
                        'teacher_final_logits', 'legal_action_mask', 'source_row', 'hidden_before',
                        'hidden_after', 'before_frames', 'after_observation_frames', 'native_seeds',
                        'episode_starts', 'dones', 'stage')}
            label = cache[number]
            z, p, legal = [label[key][offset] for key in
                ('teacher_final_logits', 'teacher_probabilities', 'legal_action_mask')]
            if (int(label['source_row'][offset]) != row or z.dtype != np.float32
                    or p.dtype != np.float32 or legal.dtype != np.bool_
                    or z.shape != (19,) or not legal[:18].all()
                    or bool(legal[18]) != bool(f[row, 4] > 0)
                    or not np.array_equal(np.isfinite(z), legal)
                    or np.any(np.abs(z[legal]) > 80) or not np.isfinite(p).all()
                    or np.any(p < 0) or np.any(p[~legal] != 0)):
                raise ValueError('Final19 distribution or inventory support differs')
            expected = np.zeros(19, np.float64)
            expected[legal] = np.exp(z[legal].astype(np.float64) - float(z[legal].max()))
            expected /= expected.sum()
            if not np.allclose(p, expected, atol=1e-6, rtol=0):
                raise ValueError('Final probabilities do not reproduce saved final logits')
            for key in ('before_frames', 'after_observation_frames', 'native_seeds',
                        'episode_starts', 'dones', 'stage'):
                if label[key][offset] != arrays[key][row]:
                    raise ValueError('Label metadata differs from independent source')
            if usable[row]:
                critical.append(dict(episode=name, source_row=row, native_seed=receipt['native_seed'],
                    split=receipt['original_BC_split'], remaining_frames=float(f[row, 19] * 8),
                    player_circle=float(f[row, 13] * 64), Bomb_probability=float(p[18]),
                    teacher_argmax=int(p.argmax())))
        if (np.any(cache[0]['hidden_before'] != 0)
                or not bool(cache[(n - 1) // 32]['dones'][-1])):
            raise ValueError('Whole source episode boundaries differ')
        total += n
        enlarged_total += int(giant.sum())
        sampled_rows += len(chosen)
        sampled_shards += len(cache)
        episodes.append(dict(name=name, native_seed=receipt['native_seed'],
            original_BC_split=receipt['original_BC_split'], rows=n,
            usable_deathbomb_rows=int(usable.sum()), enlarged_player_rows=int(giant.sum()),
            sampled_rows=len(chosen), sampled_label_shards=len(cache)))
    if len(episodes) != 30 or total != 741839 or len(critical) != 25:
        raise ValueError('Source cohort or rare-state coverage differs from the complete audit')
    write_json(output, dict(schema='B1-final19-label-bank-stratified-independent-audit-v1',
        utc=utc_now(), status='passed', bank=str(bank), bank_identity_sha256=identity_sha,
        complete_source_episodes=30, source_rows=total, enlarged_source_rows=enlarged_total,
        selected_label_rows=sampled_rows, selected_label_shards=sampled_shards,
        all25_usable_deathbomb_labels_checked=True, critical_labels=critical, episodes=episodes,
        critical_Bomb_argmax_rows=sum(row['teacher_argmax'] == 18 for row in critical),
        critical_mean_Bomb_probability=float(np.mean([row['Bomb_probability'] for row in critical])),
        source_behaviour_is_B1=False, inherited_values_used_as_truth=False,
        verification_scope='All complete episode metadata; stratified labels, including every usable deathbomb row. Not all741839 labels independently reread.',
        new_models_loaded=0, optimizer_updates=0, native_worlds_started=0,
        student_gameplay_measured=False))
    print(json.dumps(dict(status='passed', source_rows=total, episodes=30,
        selected_label_rows=sampled_rows, all25_usable_deathbomb_labels_checked=True,
        critical_Bomb_argmax_rows=sum(row['teacher_argmax'] == 18 for row in critical))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--numeric', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.bank, args.numeric, args.output)
