"""Explicit typed loading of complete SNN parameter points; no job on import."""
from pathlib import Path
import hashlib


def load_spiking_parameters(path, connectome, *, expected_sha256, device='cuda:0'):
    import torch
    from .spiking_engineered_control import SCHEMA, build_spiking_model
    from .thrl_dodge_diagnostics import _shared_binary
    def read(p, expected):
        with _shared_binary(p) as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('Immutable SNN parameter/source identity differs')
        with _shared_binary(p) as stream:
            return torch.load(stream,map_location='cpu',weights_only=True)
    payload=read(path,expected_sha256)
    if payload.get('schema')!=SCHEMA or not payload.get('actual_binary_spikes'):
        raise ValueError('Typed hard-spiking family parameter point required')
    if payload.get('raw_risk_or_teacher_action_bypass') is not False:
        raise ValueError('SNN controller may read actions/risk only from descending neurons')
    cfg=payload['core']
    if (cfg['n_neurons'],cfg['n_edges'],cfg['input_ports'],cfg['output_ports'])!=(138639,15091983,8053,1299):
        raise ValueError('Complete pinned brain and all original ports required')
    source=read(Path(payload['base_source']),payload['base_sha256'])
    model=build_spiking_model(source,connectome,seed=payload['seed'],device=device)
    model.load_state_dict(payload['state_dict'],strict=True)
    if model.core.graph_provenance['graph_fingerprint']!=cfg['graph']['graph_fingerprint']:
        raise ValueError('Signed directed graph identity changed')
    return model.eval().requires_grad_(False),payload
