"""Read saved rollout observations without materializing recurrent brain states.

For this project's local torch ZIP tapes,including losslessly compressed ones.
This is an analysis reader,not a resumable training-checkpoint loader.
"""
from pathlib import Path
import time
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def load_observations(path,*,skip_tensors=()):
    path=Path(path);started=time.monotonic()
    with FakeTensorMode():
        metadata=torch.load(path,map_location='cpu',weights_only=False)
    metadata.pop('block_states',None)
    skipped=[key for key in skip_tensors if key in metadata]
    for key in skipped:metadata.pop(key)
    archive=torch._C.PyTorchFileReader(str(path))
    records={archive.get_record_offset(name):name for name in archive.get_all_records() if name.startswith('data/')}
    storages={};loaded_bytes=0

    def materialize(value):
        nonlocal loaded_bytes
        if torch.is_tensor(value):
            storage=value.untyped_storage();offset=storage._checkpoint_offset
            key=(offset,value.dtype)
            if key not in storages:
                if storage.nbytes():
                    content=bytearray(archive.get_record(records[offset]));loaded_bytes+=len(content)
                    storages[key]=torch.frombuffer(content,dtype=value.dtype)
                else:storages[key]=torch.empty(0,dtype=value.dtype)
            return torch.as_strided(storages[key],tuple(value.shape),value.stride(),value.storage_offset())
        if isinstance(value,dict):return {key:materialize(item) for key,item in value.items()}
        if isinstance(value,list):return [materialize(item) for item in value]
        if isinstance(value,tuple):return tuple(materialize(item) for item in value)
        return value

    result=materialize(metadata)
    return result,dict(method='torch_fake_metadata_then_observation_storage_only',
        actual_storage_bytes_read=loaded_bytes,actual_storage_records_read=len(storages),
        bytes_measure='Uncompressed tensor storage bytes materialized;excludes ZIP/pickle metadata',
        skipped_tensor_fields=skipped,brain_state_payload_loaded=False,seconds=time.monotonic()-started)
