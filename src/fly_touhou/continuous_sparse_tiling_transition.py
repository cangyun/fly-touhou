"""NoTorch metadata-only transition for effective nested sparse configurations.

Does not load/save a checkpoint or touch tensor data. The caller owns later
roundtrip/typed loading after a chunk has actually been selected by measurement.
"""
from __future__ import annotations

SCHEMA='continuous-sparse-runtime-tiling-transition-v1'
MEASUREMENT_CANDIDATES=(65536,262144,1048576,4194304)


def prepare_runtime_tiling_transition(source,edge_chunk_size,*,expected_old_chunk_size=65536):
    """Return(new_training_payload,record), preserving old tensors by reference.

    Only source['model_payload'] is visited. A match is exactly a dict path
    ``*.configuration.core.sparse.edge_chunk_size``. Algorithm/provenance trees
    are never visited/rewritten, including if present inside model metadata.
    No optimizer/RNG/counters/named tensor or physical model setting changes.
    """
    if not isinstance(source,dict) or not isinstance(source.get('model_payload'),dict):
        raise TypeError('Explicit complete training payload with model_payload required')
    if type(edge_chunk_size) is not int or edge_chunk_size not in MEASUREMENT_CANDIDATES:
        raise ValueError('Choose an explicitly measured tiling candidate')
    if type(expected_old_chunk_size) is not int or expected_old_chunk_size<1:
        raise ValueError('Explicit positive old tiling expected')
    changed=[]

    def visit(value,path):
        if not isinstance(value,dict):
            return value
        result=value
        for key,child in value.items():
            if key in ('algorithm','provenance','historical_provenance') or key.endswith('_provenance'):
                continue
            if key=='configuration' and isinstance(child,dict):
                core=child.get('core')
                sparse=core.get('sparse') if isinstance(core,dict) else None
                if isinstance(sparse,dict) and 'edge_chunk_size' in sparse:
                    old=sparse['edge_chunk_size']
                    if old!=expected_old_chunk_size:
                        raise ValueError('Unexpected effective tiling at '+'.'.join((*path,key,'core','sparse','edge_chunk_size')))
                    replacement=dict(child,core=dict(core,sparse=dict(sparse,edge_chunk_size=edge_chunk_size)))
                    if result is value:
                        result=dict(value)
                    result[key]=replacement
                    changed.append(dict(path='.'.join((*path,key,'core','sparse','edge_chunk_size')),
                        old=old,new=edge_chunk_size))
                    continue
            replacement=visit(child,(*path,str(key)))
            if replacement is not child:
                if result is value:
                    result=dict(value)
                result[key]=replacement
        return result

    model=visit(source['model_payload'],('model_payload',))
    if not changed:
        raise ValueError('No effective configuration.core.sparse.edge_chunk_size found')
    record=dict(schema=SCHEMA,old_edge_chunk_size=expected_old_chunk_size,new_edge_chunk_size=edge_chunk_size,
        effective_configuration_paths=changed,changed_configuration_count=len(changed),
        original_tensors_optimizer_RNG_counters_preserved_by_reference=True,
        historical_algorithm_and_provenance_unchanged=True,
        physical_model_or_gradient_formula_or_reward_changed=False,
        bitwise_equality_claimed=False,benchmark_or_checkpoint_roundtrip_executed=False)
    if 'runtime_tiling_transition' in source:
        record['previous_runtime_tiling_transition']=source['runtime_tiling_transition']
    result=dict(source,model_payload=model,runtime_tiling_transition=record)
    return result,record
