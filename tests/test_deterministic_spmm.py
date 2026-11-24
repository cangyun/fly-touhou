import pytest
import torch

from fly_touhou.deterministic_spmm import multiply


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("batch", [1, 128])
def test_compact_csr_preserves_the_four_subsequences(batch, monkeypatch, tmp_path):
    monkeypatch.setenv("FLY_TOUHOU_SPMM_CACHE", str(tmp_path))
    rng = torch.Generator().manual_seed(1024)
    rows = 67
    counts = torch.cat((torch.tensor([0, 1, 512]), torch.randint(0, 54, (rows - 3,), generator=rng)))
    csr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))).cuda()
    columns = torch.randint(rows, (int(counts.sum()),), generator=rng).cuda()
    weights = torch.randn(len(columns), generator=rng).cuda()
    activity = torch.randn(batch, rows, generator=rng).cuda()
    expected = multiply(activity, weights, csr, columns)
    actual = multiply(activity, weights, csr, columns, compact=True)
    assert torch.equal(expected, actual)

    # 0 * inf in a valid CSR column must still be evaluated.
    activity[:, columns[1]] = 0
    weights[1] = float("inf")
    expected = multiply(activity, weights, csr, columns)
    actual = multiply(activity, weights, csr, columns, compact=True)
    assert expected.isnan().any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("batch", [32, 128])
def test_packed_training_spikes_preserve_csr_output(batch, monkeypatch, tmp_path):
    from fly_touhou.frozen_binary_messages import multiply_training_spikes

    monkeypatch.setenv("FLY_TOUHOU_SPMM_CACHE", str(tmp_path))
    rng = torch.Generator().manual_seed(217)
    counts = torch.tensor([0, 1, 33, 7, 241, 2, 4])
    rows = len(counts)
    ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))).cuda()
    columns = torch.randint(rows, (int(counts.sum()),), generator=rng).cuda()
    spikes = torch.randint(2, (batch, rows), generator=rng).float().cuda()
    weights = torch.randn(len(columns), generator=rng).cuda()
    expected = multiply(spikes, weights, ptr, columns)
    assert torch.equal(multiply_training_spikes(spikes, weights, ptr, columns), expected)

    spikes[:, columns[0]] = 0
    weights[0] = float("inf")
    expected = multiply(spikes, weights, ptr, columns)
    assert expected[:, 1].isnan().all()
    torch.testing.assert_close(multiply_training_spikes(spikes, weights, ptr, columns),
                               expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("batch", [32, 128])
def test_edge_gradient_prunes_only_finite_zero_tiles(batch, monkeypatch, tmp_path):
    from fly_touhou.fused_edge_gradient import edge_gradient

    monkeypatch.setenv("FLY_TOUHOU_EDGE_GRAD_CACHE", str(tmp_path))
    rng = torch.Generator().manual_seed(825)
    rows = 83
    pre = torch.randint(rows, (267,), generator=rng).cuda()
    post = torch.randint(rows, (267,), generator=rng).cuda()
    activity = torch.randn(batch, rows, generator=rng).cuda()
    gradient = torch.randn(batch, rows, generator=rng).cuda()
    activity[:, :20] = 0
    gradient[:, 40:60] = 0
    activity[:32, 60:] = 0
    expected = edge_gradient(activity, gradient, pre, post)
    actual = edge_gradient(activity, gradient, pre, post, prune_zero_tiles=True)
    assert torch.equal(actual, expected)

    # Two original edges produce 0 * Inf and NaN * 0, respectively.
    pre[:2] = torch.tensor([0, 81], device=activity.device)
    post[:2] = torch.tensor([82, 40], device=activity.device)
    gradient[-1, 82] = float("inf")
    activity[1, 81] = float("nan")
    expected = edge_gradient(activity, gradient, pre, post)
    actual = edge_gradient(activity, gradient, pre, post, prune_zero_tiles=True)
    assert expected[:2].isnan().all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_packed_history_preserves_fp32_weights_and_reduction(monkeypatch, tmp_path):
    from fly_touhou.deterministic_spmm import multiply_history, pack_history_edges

    monkeypatch.setenv("FLY_TOUHOU_SPMM_CACHE", str(tmp_path))
    rng = torch.Generator().manual_seed(381)
    counts = torch.tensor([0, 1, 4, 33, 127, 257, 9, 3])
    rows = len(counts)
    crow = torch.cat((torch.zeros(1,dtype=torch.long), counts.cumsum(0))).cuda()
    columns = torch.randint(rows, (int(counts.sum()),), generator=rng).cuda()
    weights = torch.randn(len(columns), generator=rng).cuda()
    activity = torch.randn(1, rows, generator=rng).cuda()
    assert torch.equal(multiply_history(activity, crow, pack_history_edges(columns, weights)),
                       multiply(activity, weights, crow, columns, compact=True))
    weights[0] = float("inf")
    activity[:, columns[0]] = 0
    expected = multiply(activity, weights, crow, columns, compact=True)
    got = multiply_history(activity, crow, pack_history_edges(columns, weights))
    assert expected[0, 1].isnan()
    torch.testing.assert_close(got, expected, rtol=0., atol=0., equal_nan=True)
