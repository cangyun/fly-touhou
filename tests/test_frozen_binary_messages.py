from types import SimpleNamespace

import pytest
import torch

from fly_touhou.deterministic_spmm import multiply
from fly_touhou.frozen_binary_messages import FrozenBinaryMessages, _kernels


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device required')
def test_fixed_order_spike_sums_match_across_candidate_groups(tmp_path):
    generator=torch.Generator().manual_seed(729)
    counts=torch.tensor([0,1,3,32,33,129,241,7,13,72],dtype=torch.int64)
    ptr=torch.cat((torch.zeros(1,dtype=torch.int64),counts.cumsum(0))).cuda()
    columns=torch.randint(len(counts),(int(counts.sum()),),generator=generator).cuda()
    weights=torch.randn(len(columns),generator=generator).cuda()
    kernel=object.__new__(FrozenBinaryMessages)
    kernel.core=SimpleNamespace(sparse=SimpleNamespace(forward_crow=ptr,forward_columns=columns))
    kernel.ordered=weights
    kernel.pack,kernel.reduce=_kernels(weights.device,tmp_path)
    for batch in (1,18,32,36,128):
        spikes=torch.randint(2,(batch,len(counts)),generator=generator).float().cuda()
        assert torch.equal(kernel.message(spikes,weights),multiply(spikes,weights,ptr,columns))
