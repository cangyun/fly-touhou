"""Compare optimized and reference native corridors on archived game snapshots."""
import argparse
import json
import pickle
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def assert_equal(before, after, field=""):
    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        np.testing.assert_array_equal(before, after, err_msg=field)
    elif isinstance(before, dict):
        if before.keys() != after.keys():
            raise ValueError((field, before.keys(), after.keys()))
        for key in before:
            assert_equal(before[key], after[key], f"{field}.{key}")
    elif isinstance(before, (list, tuple)):
        if len(before) != len(after):
            raise ValueError((field, len(before), len(after)))
        for index, (left, right) in enumerate(zip(before, after)):
            assert_equal(left, right, f"{field}[{index}]")
    elif isinstance(before, (float, np.floating)) and np.isnan(before) and np.isnan(after):
        return
    elif before != after:
        raise ValueError((field, before, after))


def run(repo, scenes, output, optimization="aim"):
    sys.path.insert(0, str(repo / "src"))
    from fly_touhou import native_ex_conditional as cond, native_curve_motion_paths, native_wall_geometry
    from fly_touhou import native_bullet_release as release
    from fly_touhou.nbnm_corridor import plan
    original = cond._waiting_forecasts
    packed = cond._waiting_clearances
    sweep_key = release._sweep_key

    def general(*args, **kwargs):
        return None

    def unique_key(*args, **kwargs):
        return object()

    results = []
    for path in sorted(scenes.glob("s*-*.pkl")):
        data = path.read_bytes()
        def measure(optimized):
            use_aim = optimized or optimization == "tables"
            use_tables = optimized or optimization == "aim"
            cond._waiting_forecasts = original if use_aim else general
            cond._waiting_clearances = packed if use_aim else general
            release._sweep_key = sweep_key if use_tables else unique_key
            cond._CACHE.clear()
            native_curve_motion_paths._NODE_CACHE.clear()
            native_wall_geometry._cached_positions.cache_clear()
            encoding, args, kwargs = pickle.loads(data)
            start = perf_counter()
            result = plan(encoding, *args, **kwargs)
            return result, 1000 * (perf_counter() - start)
        try:
            baseline, _ = measure(False)
            tested, _ = measure(True)
            assert_equal(baseline, tested, path.stem)
            variant_rows = []
            for first in (False, True, False):
                _, t = measure(first)
                _, u = measure(not first)
                variant_rows.append((u, t) if first else (t, u))
            a, b = np.median(variant_rows, axis=0)
            results.append(dict(scene=path.stem, baseline_ms=a, optimized_ms=b,
                                equal_numeric_outputs=True, repeat_pairs_ms=variant_rows))
            output.write_text(json.dumps(dict(optimization=optimization, scenes=results), indent=2), encoding="utf-8")
            print(path.stem, round(a, 2), round(b, 2), flush=True)
        finally:
            cond._waiting_forecasts = original
            cond._waiting_clearances = packed
            release._sweep_key = sweep_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--optimization", choices=("aim", "tables", "combined"), default="aim")
    args = parser.parse_args()
    run(args.repo, args.scenes, args.output, args.optimization)
