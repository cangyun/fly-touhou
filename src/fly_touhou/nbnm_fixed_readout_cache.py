"""Persist immutable prefix states for a separate readout-only update phase."""
from pathlib import Path
from pickle import UnpicklingError


def identity(model, spec, checkpoint, histories):
    import torch

    paths = {Path(checkpoint).resolve(), Path(spec['environment_config']).resolve()}
    paths.update(Path(name).resolve() for name, _ in histories)
    source_dir=Path(__file__).resolve().parent
    paths.update(source_dir / name for name in ('nbnm_prefix.py', 'nbnm_sensation.py', 'compact_story_model.py',
                 'spiking_engineered_control.py', 'continuous_sparse.py', 'deterministic_spmm.py'))
    versions = []
    for path in sorted(paths):
        info = path.stat()
        versions.append((str(path), info.st_size, info.st_mtime_ns))
    capability = torch.cuda.get_device_capability(model.device) if model.device.type == 'cuda' else None
    return dict(schema='nbnm-fixed-readout-prefix-v1', sources=versions, windows=sorted(histories),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda, capability=capability,
                history_cuda_graph=bool(model.nbnm.get('history_cuda_graph')))


def _sources_by_kind(signature):
    history_paths = {str(Path(name).resolve()) for name, _ in signature['windows']}
    model_sources, history_sources = {}, {}
    for path, size, modified in signature['sources']:
        selected = history_sources if path in history_paths else model_sources
        selected[path] = size, modified
    return model_sources, history_sources


def _matching_windows(actual, expected, *, allow_subset):
    if actual == expected:
        return set(expected['windows'])
    if (not allow_subset or not isinstance(actual, dict) or actual.keys() != expected.keys()
            or any(actual[key] != value for key, value in expected.items()
                   if key not in ('windows', 'sources'))):
        return None
    old_model, old_histories = _sources_by_kind(actual)
    new_model, new_histories = _sources_by_kind(expected)
    if old_model != new_model:
        return None
    matching = set(expected['windows']).intersection(actual['windows'])
    return {key for key in matching if old_histories[str(Path(key[0]).resolve())]
            == new_histories[str(Path(key[0]).resolve())]}


def read_states(path, expected, model, *, allow_subset=False):
    """Allow the normal replay path to handle absent or stale prepared states."""
    import torch

    if not path.is_file():
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except (OSError, RuntimeError, EOFError, ValueError, UnpicklingError):
        return None
    if not isinstance(data, dict):
        return None
    stored = data.get('identity')
    keys = _matching_windows(stored, expected, allow_subset=allow_subset)
    if keys is None:
        return None
    snapshots = data.get('states')
    if not isinstance(snapshots, dict) or set(snapshots) != set(stored['windows']):
        return None
    template = model.initial_state(1)
    fields = template.tensors()
    for key in keys:
        state = snapshots[key]
        if (not isinstance(state, (tuple, list)) or len(state) != len(fields)
                or any(not torch.is_tensor(value) or value.shape != field.shape or value.dtype != field.dtype
                       for value, field in zip(state, fields, strict=True))):
            return None
    return {key: type(template)(*(value.to(model.device) for value in snapshots[key])) for key in keys}
