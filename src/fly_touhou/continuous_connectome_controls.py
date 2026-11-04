"""CPU topology control preserving every stored edge and its sign/magnitude.

Permute target rows only, within identical indegree bins, and protect sensory
and output rows. This changes wiring rather than consistently renaming both
ends of every edge. No duplicate merging, new connections or sparse->dense
matrix is used. Apply AFTER the measured graph's normalization; do not then
renormalize the control. The control is never substituted for the main graph.
"""
from __future__ import annotations

import hashlib
import numpy as np


def degree_matched_target_control(crow,col,values,*,protected_nodes=(),seed=82931):
    crow=np.asarray(crow);col=np.asarray(col);values=np.asarray(values)
    if (crow.ndim!=1 or col.ndim!=1 or values.ndim!=1 or len(crow)<2
            or len(col)!=len(values) or crow[0]!=0 or crow[-1]!=len(col)
            or crow.dtype.kind not in 'iu' or col.dtype.kind not in 'iu'
            or not np.isfinite(values).all() or np.any(values==0)):
        raise ValueError('Finite nonzero original CSR weights and integer structure required')
    n=len(crow)-1;degrees=np.diff(crow)
    if np.any(degrees<0) or np.any(col<0) or np.any(col>=n):raise ValueError('Invalid CSR indices')
    protected=np.asarray(protected_nodes,dtype=np.int64)
    if np.any(protected<0) or np.any(protected>=n):raise ValueError('Protected IDs must be graph indices')
    mask=np.zeros(n,bool);mask[protected]=True
    eligible=np.flatnonzero(~mask)
    generator=np.random.default_rng(seed)
    target_for_old=np.arange(n,dtype=np.int64)
    # Bins use unweighted indegree. Original edge values travel with edges;
    # weighted input totals may move and this distinction is explicit below.
    for degree in np.unique(degrees[eligible]):
        group=eligible[degrees[eligible]==degree]
        if len(group)>1:target_for_old[group]=generator.permutation(group)
    old_for_new=np.empty(n,dtype=np.int64);old_for_new[target_for_old]=np.arange(n)
    new_rows=np.repeat(np.arange(n,dtype=np.int64),degrees)
    offset=np.arange(len(col),dtype=np.int64)-crow[new_rows]
    source_edge=crow[old_for_new[new_rows]]+offset
    new_col=col[source_edge].copy();new_values=values[source_edge].copy()
    if not np.array_equal(np.bincount(new_col,minlength=n),np.bincount(col,minlength=n)):
        raise RuntimeError('Source outdegrees unexpectedly changed')
    digest=hashlib.sha256()
    for item in (crow,new_col,new_values):digest.update(np.ascontiguousarray(item).tobytes())
    return dict(crow=crow.copy(),col=new_col,values=new_values,source_edge_permutation=source_edge,
        target_for_old=target_for_old,metadata=dict(schema='continuous-degree-matched-target-row-control-v1',
            seed=seed,nodes=n,edges=len(col),protected_nodes=int(mask.sum()),
            moved_nonempty_rows=int(((target_for_old!=np.arange(n))&(degrees>0)).sum()),
            changed_stored_entries=int(((new_col!=col)|(new_values!=values)).sum()),
            graph_sha256=digest.hexdigest(),node_identities_and_ports_fixed=True,
            all_directed_edge_entries_preserved=True,unique_pairs_preserved_if_source_unique=True,
            source_outdegree_and_target_indegree_exact=True,individual_edge_values_and_signs_preserved=True,
            source_columns_not_permuted=True,whole_node_relabeling=False,
            weighted_target_totals_permuted_with_rows=True,normalization_not_recomputed=True,
            main_model_graph_unchanged=True))
