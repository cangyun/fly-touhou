"""First-order trainable directed sparse SpMM without an N-by-N gradient.

W[post,pre] uses the original edge order. Forward/gradX use pre-indexed CSR
matrices; gradEdge is chunked SDDMM sum_batch(gradY[post]*X[pre]). The custom
autograd boundary never asks PyTorch for a sparse-matrix-value backward, which
can materialize a dense square matrix. No old fixed cuSPARSE source is changed.
"""
from __future__ import annotations

import numpy as np
import os
import torch
from torch import nn
from torch.autograd.function import once_differentiable

SCHEMA = 'continuous-directed-CSR-SpMM-chunked-SDDMM-v1'


class _SparseEdgeMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activity, edge_values, operator):
        ctx.operator = operator
        ctx.save_for_backward(activity, edge_values)
        return operator._multiply(activity, edge_values, transpose=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        activity, edge_values = ctx.saved_tensors
        operator = ctx.operator
        grad_activity = (operator._multiply(grad_output, edge_values, transpose=True)
            if ctx.needs_input_grad[0] else None)
        grad_edges = None
        if ctx.needs_input_grad[1]:
            if operator.fused_edge_grad and activity.is_cuda:
                from .fused_edge_gradient import edge_gradient
                grad_edges=edge_gradient(activity,grad_output,operator.pre,operator.post)
            else:
                grad_edges = torch.empty_like(edge_values)
                for start in range(0, operator.n_edges, operator.edge_chunk_size):
                    end = min(start + operator.edge_chunk_size, operator.n_edges)
                    # Only B*chunk temporary elements, never N*N or E*B at once.
                    outgoing = activity.index_select(1, operator.pre[start:end])
                    incoming_gradient = grad_output.index_select(1, operator.post[start:end])
                    grad_edges[start:end] = (incoming_gradient * outgoing).sum(0)
        return grad_activity, grad_edges, None


class TrainableDirectedSparse(nn.Module):
    """Immutable sparse structure; differentiable values remain caller-owned."""
    def __init__(self, n_neurons, pre, post, *, edge_chunk_size=65536):
        super().__init__()
        if type(n_neurons) is not int or not 0 < n_neurons < 2**31:
            raise ValueError('Positive int32-sized node count required')
        if type(edge_chunk_size) is not int or edge_chunk_size < 1:
            raise ValueError('Positive finite SDDMM edge chunk size required')
        pre, post = np.asarray(pre), np.asarray(post)
        if (pre.ndim != 1 or post.shape != pre.shape or pre.dtype.kind not in 'iu'
                or post.dtype.kind not in 'iu' or len(pre) < 1
                or min(int(pre.min()),int(post.min())) < 0
                or max(int(pre.max()),int(post.max())) >= n_neurons):
            raise ValueError('Original in-range directed edge arrays required')
        self.n_neurons, self.n_edges = n_neurons, len(pre)
        self.edge_chunk_size = edge_chunk_size
        self.fused_edge_grad=os.environ.get('FLY_TOUHOU_FUSED_EDGE_GRAD','0')=='1'
        self.deterministic_spmm=os.environ.get('FLY_TOUHOU_DETERMINISTIC_SPMM','0')=='1'
        pre, post = pre.astype(np.int64,copy=False), post.astype(np.int64,copy=False)
        self.register_buffer('pre', torch.from_numpy(pre.copy()), persistent=False)
        self.register_buffer('post', torch.from_numpy(post.copy()), persistent=False)
        # Sorting is done once on CPU. Neither directed edges nor duplicate
        # source entries are coalesced, removed, invented or reordered as parameters.
        for prefix, rows, columns in (('forward',post,pre),('transpose',pre,post)):
            permutation = np.argsort(rows*n_neurons+columns, kind='stable')
            counts = np.bincount(rows, minlength=n_neurons)
            crow = np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(counts,dtype=np.int64)))
            self.register_buffer(prefix+'_permutation',torch.from_numpy(permutation),persistent=False)
            self.register_buffer(prefix+'_crow',torch.from_numpy(crow),persistent=False)
            self.register_buffer(prefix+'_columns',torch.from_numpy(columns[permutation].copy()),persistent=False)

    def _multiply(self, activity, values, *, transpose):
        prefix = 'transpose' if transpose else 'forward'
        if self.deterministic_spmm and activity.is_cuda:
            from .deterministic_spmm import multiply
            return multiply(activity,values.index_select(0,getattr(self,prefix+'_permutation')),
                getattr(self,prefix+'_crow'),getattr(self,prefix+'_columns'))
        matrix = torch.sparse_csr_tensor(getattr(self,prefix+'_crow'),getattr(self,prefix+'_columns'),
            values.index_select(0,getattr(self,prefix+'_permutation')),
            size=(self.n_neurons,self.n_neurons),dtype=values.dtype,device=values.device)
        return torch.sparse.mm(matrix,activity.T.contiguous()).T.contiguous()

    def forward(self, activity, edge_values):
        if (activity.ndim != 2 or activity.shape[0] < 1 or activity.shape[1] != self.n_neurons
                or edge_values.shape != (self.n_edges,) or activity.dtype != torch.float32
                or edge_values.dtype != torch.float32 or activity.device != edge_values.device
                or activity.device != self.pre.device):
            raise ValueError('Same-device FP32 activity[B,N] and original-order values[E] required')
        return _SparseEdgeMM.apply(activity,edge_values,self)

    def config(self):
        return dict(schema=SCHEMA,n_neurons=self.n_neurons,n_edges=self.n_edges,
            backend='fixed-order FP32 CUDA CSR; custom first-order backward' if self.deterministic_spmm else 'torch CSR sparse.mm; custom first-order backward',
            edge_chunk_size=self.edge_chunk_size,grad_activity='transposed original directed graph',
            grad_edge='optional fused CUDA original-order SDDMM' if self.fused_edge_grad else 'chunked sum_batch(gradY[post]*X[pre])',
            square_dense_matrix_or_gradient_created=False,temporary_shape='at most[B,edge_chunk_size]',
            source_edge_order_preserved=True,structure_buffers_reconstructed_not_checkpointed=True,
            deterministic_spmm=self.deterministic_spmm,
            higher_order_gradients_supported=False,requires_bitwise_deterministic_runtime=False)
