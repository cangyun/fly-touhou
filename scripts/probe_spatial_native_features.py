"""Offline sensor qualification using existing training snapshots, never tests."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch
from gymnasium import spaces
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.spatial_native_features import SpatialNativeFeatures
from fly_touhou.standard_gru_ppo import NativeFeatures
from fly_touhou.state import Snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--training-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    encoder = GeometryObservationEncoder.from_specification(protocol['encoder'])
    observations, context = [], []
    with gzip.open(args.training_source, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['role'] != 'training':
                raise ValueError('This probe cannot consume held-out test records')
            snapshot = Snapshot.from_dict(row['snapshot'])
            p = snapshot.player
            nearby = [b for b in snapshot.bullets if b.active and not b.intangible
                      and abs(b.x-p.x) < 85 and abs(b.y-p.y) < 85]
            if not nearby:
                continue
            observations.append(encoder.encode(snapshot))
            context.append({'episode': row['episode_id'], 'frame': snapshot.frame,
                            'nearby_active': len(nearby), 'total_bullets': len(snapshot.bullets)})
            if len(observations) == 64:
                break
    if len(observations) < 8:
        raise ValueError('Need at least8 actual training states with nearby native bullets')
    tensor = torch.as_tensor(np.stack(observations), device='cuda')
    p = encoder.planner.config
    kwargs = {'feature_names': encoder.feature_names, 'bounds': (p.left, p.right, p.top, p.bottom)}
    box = spaces.Box(-1., 1., shape=(1911,), dtype=np.float32)
    spatial = SpatialNativeFeatures(box, **kwargs).cuda().eval()
    flat = NativeFeatures(box).cuda().eval()
    with torch.no_grad():
        grids = spatial.rasterize(tensor)
        features = spatial(tensor)
        assert grids.shape == (len(observations), 8, 48, 48)
        assert torch.isfinite(grids).all() and torch.isfinite(features).all()
        occupied = grids[:, :2].amax(1).sum((1, 2))
        assert (occupied > 0).all()
        latencies = {}
        for label, model in [('flat', flat), ('spatial', spatial)]:
            batch = tensor[:4]
            for _ in range(10):
                model(batch)
            torch.cuda.synchronize()
            elapsed = []
            for _ in range(31):
                started = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-started)*1000)
            latencies[label] = {'batch4_median_ms': float(np.median(elapsed)),
                                'batch4_p95_ms': float(np.quantile(elapsed, .95))}
    np.savez_compressed(output/'training_observations_and_grids.npz',
                        observations=np.stack(observations), grids=grids.cpu().numpy())
    result = {'schema': 'offline-current-state-spatial-qualification-v1',
        'native_game_started': False, 'native_actions': 0, 'heldout_records': 0,
        'source': str(args.training_source.resolve()),
        'source_sha256': hashlib.sha256(args.training_source.read_bytes()).hexdigest(),
        'native_training_states': len(observations), 'context': context,
        'all_nearby_bullet_states_have_nonempty_occupancy': bool((occupied > 0).all()),
        'channels': spatial.CHANNELS, 'grid_shape': list(grids.shape),
        'information': 'only existing raw1911; retained raw branch; same finite object slots',
        'predictions_or_action_scores': False, 'learning_success': False,
        'feature_parameters': {'flat': sum(p.numel() for p in flat.parameters()),
                               'spatial': sum(p.numel() for p in spatial.parameters())},
        'shared_GPU_feature_latency': latencies,
        'source_code_sha256': hashlib.sha256((ROOT/'src/fly_touhou/spatial_native_features.py').read_bytes()).hexdigest()}
    (output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'context'}, ensure_ascii=False))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
