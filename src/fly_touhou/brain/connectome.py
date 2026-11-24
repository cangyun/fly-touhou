"""Load the actual FlyWire v783 graph without changing its neuron indexing.

The first column of Completeness_783.csv is the FlyWire root ID.  The row
number, *not* the sorted ID, is the index used by Connectivity_783.parquet.
The upstream signed strength column already includes the predicted synaptic
sign.  We retain that column verbatim; a synapse count is not a fitted
physiological conductance.

The default experiment is an induced, weakly connected 2,048-neuron subset.
Its selection is topological and must not be described as an identified visual
or motor circuit. Pass max_neurons=None to load all 138,639 neurons in these
upstream model files. The atlas publication's 139,255-neuron count is a
different figure; missing neurons are never padded or invented here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

SOURCE_REPOSITORY = "https://github.com/philshiu/Drosophila_brain_model"
SOURCE_REVISION = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SOURCE_RAW_BASE = (
    "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
    + SOURCE_REVISION
)
EXPECTED_NEURONS = 138_639
PUBLISHED_ATLAS_NEURONS = 139_255
# Recorded from the bytes served by the immutable upstream revision above.
SOURCE_FILES = {
    "Completeness_783.csv": {
        "sha256": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
        "size_bytes": 3_327_347,
    },
    "Connectivity_783.parquet": {
        "sha256": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        "size_bytes": 100_804_642,
    },
    "LICENSE": {
        "sha256": "3621f6d6476189190e2960fa43f11b275ba4eca848fdac50a1ba2925de6223e8",
        "size_bytes": 1_085,
    },
    "model.py": {
        "sha256": "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e",
        "size_bytes": 11_900,
    },
}
PRE_COLUMN = "Presynaptic_Index"
POST_COLUMN = "Postsynaptic_Index"
WEIGHT_COLUMN = "Excitatory x Connectivity"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_files(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Check both graph files against pinned, known hashes; never download here."""
    directory = Path(directory)
    result = {}
    for filename in ("Completeness_783.csv", "Connectivity_783.parquet"):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/fetch_connectome.py first."
            )
        actual = file_sha256(path)
        expected = SOURCE_FILES[filename]["sha256"]
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
        result[filename] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return result


@dataclass
class ConnectomeGraph:
    """Compact directed graph; rows receive spikes and columns send them.

    ``pre``, ``post`` index this graph, while ``source_indices`` preserves the
    upstream row mapping. ``node_ids`` are decimal strings to prevent precision
    loss in JSON or libraries which would otherwise cast IDs to floating point.
    ``signed_weights`` retain upstream units (signed connection count).
    """

    node_ids: tuple[str, ...]
    source_indices: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    signed_weights: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.isdecimal() for value in self.node_ids):
            raise ValueError("FlyWire IDs must be exact decimal strings, never floats")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("Duplicate FlyWire root IDs")
        self.source_indices = _integer_array(self.source_indices, "source_indices")
        self.pre = _integer_array(self.pre, "pre")
        self.post = _integer_array(self.post, "post")
        self.signed_weights = np.asarray(self.signed_weights, dtype=np.float32)
        if self.signed_weights.ndim != 1:
            raise ValueError("signed_weights must have one dimension")
        if len(self.source_indices) != self.n_neurons:
            raise ValueError("source_indices must correspond one-to-one with node_ids")
        if len(set(self.source_indices.tolist())) != self.n_neurons:
            raise ValueError("Duplicate source indices")
        if len(self.pre) != len(self.post) or len(self.pre) != len(self.signed_weights):
            raise ValueError("Edge columns differ in length")
        if not np.isfinite(self.signed_weights).all():
            raise ValueError("Non-finite connection strength")
        if self.n_neurons < 1:
            raise ValueError("A graph must contain neurons")
        if len(self.pre) and (
            min(int(self.pre.min()), int(self.post.min())) < 0
            or max(int(self.pre.max()), int(self.post.max())) >= self.n_neurons
        ):
            raise ValueError("Edge index outside graph")

    @property
    def n_neurons(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    @property
    def weights(self) -> np.ndarray:
        """Alias for callers which explicitly want the original signed counts."""
        return self.signed_weights

    def incoming_matrix(self, *, normalize: bool = True, gain: float = 1.0) -> sparse.csr_matrix:
        """Return W[post, pre]; optionally normalize each incoming absolute sum.

        Normalization is an engineering choice for a stable frozen reservoir;
        it preserves the edge signs but changes physiological magnitudes.
        """
        matrix = sparse.coo_matrix(
            (self.signed_weights, (self.post, self.pre)),
            shape=(self.n_neurons, self.n_neurons),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if normalize:
            incoming = np.asarray(abs(matrix).sum(axis=1)).reshape(-1)
            inverse = np.divide(
                1.0, incoming, out=np.zeros_like(incoming), where=incoming > 0
            )
            matrix = sparse.diags(inverse) @ matrix
        matrix.data *= np.float32(gain)
        return matrix.tocsr()

    def summary(self) -> dict[str, Any]:
        return {
            "neurons": self.n_neurons,
            "directed_edges": self.n_edges,
            "positive_edges": int((self.signed_weights > 0).sum()),
            "negative_edges": int((self.signed_weights < 0).sum()),
            "zero_edges": int((self.signed_weights == 0).sum()),
            "absolute_connection_count": float(np.abs(self.signed_weights).sum(dtype=np.float64)),
            "source_indices": self.source_indices.tolist(),
            "flywire_root_ids": list(self.node_ids),
            "provenance": self.provenance,
        }


def _integer_array(values: np.ndarray | Iterable[int], label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    if len(array) and (array.min() < 0 or array.max() > np.iinfo(np.int32).max):
        raise ValueError(f"{label} contains unsupported indices")
    return array.astype(np.int32, copy=False)


def _connected_selection(
    pre: np.ndarray,
    post: np.ndarray,
    strengths: np.ndarray,
    n_neurons: int,
    size: int,
    seed_index: int | None,
) -> tuple[np.ndarray, int]:
    """Deterministic strongest-neighbor BFS over the weak undirected graph."""
    if not 1 <= size <= n_neurons:
        raise ValueError(f"max_neurons must be between 1 and {n_neurons}")
    adjacency = sparse.coo_matrix(
        (np.abs(strengths), (pre, post)), shape=(n_neurons, n_neurons)
    ).tocsr()
    adjacency = adjacency + adjacency.T
    adjacency.eliminate_zeros()
    if seed_index is None:
        incident_strength = np.asarray(adjacency.sum(axis=1)).ravel()
        seed_index = int(np.argmax(incident_strength))
    if not 0 <= seed_index < n_neurons:
        raise ValueError("seed_index outside the source graph")
    selected = np.zeros(n_neurons, dtype=bool)
    selected[seed_index] = True
    queue = deque([seed_index])
    count = 1
    while queue and count < size:
        node = queue.popleft()
        start, stop = adjacency.indptr[node : node + 2]
        neighbors = adjacency.indices[start:stop]
        weights = adjacency.data[start:stop]
        # Primary key: strongest first. Tie breaker: original index ascending.
        ordering = np.lexsort((neighbors, -weights))
        for neighbor in neighbors[ordering]:
            if not selected[neighbor]:
                selected[neighbor] = True
                queue.append(int(neighbor))
                count += 1
                if count == size:
                    break
    if count != size:
        raise ValueError(
            f"Seed {seed_index} has a connected component of only {count} neurons; "
            "choose a smaller max_neurons or a different seed_index"
        )
    # Retain the upstream relative order in the compact representation.
    return np.flatnonzero(selected).astype(np.int32), seed_index


def load_connectome(
    directory: str | Path,
    max_neurons: int | None = 2048,
    seed_index: int | None = None,
    *,
    source_indices: Iterable[int] | None = None,
    verify: bool = True,
) -> ConnectomeGraph:
    """Read v783 (full graph or a real induced subset), preserving signed edges.

    Pass saved ``source_indices`` to reproduce a trained student's exact graph.
    Passing max_neurons=None uses the full graph. Hash verification is enabled
    by default; turning it off is useful only for deliberately constructed
    schema tests and does not confer verified-data provenance.
    """
    directory = Path(directory)
    checked = verify_source_files(directory) if verify else {}
    # Do not permit pandas to infer an ID float dtype, even with missing values.
    completeness = pd.read_csv(directory / "Completeness_783.csv", dtype=str)
    ids = tuple(completeness.iloc[:, 0].tolist())
    if any(not isinstance(value, str) or not value.isdecimal() for value in ids):
        raise ValueError("Invalid root ID in Completeness_783.csv")
    n_source = len(ids)
    if verify and n_source != EXPECTED_NEURONS:
        raise ValueError(f"Expected {EXPECTED_NEURONS} v783 neurons; found {n_source}")
    table = pq.read_table(
        directory / "Connectivity_783.parquet",
        columns=[PRE_COLUMN, POST_COLUMN, WEIGHT_COLUMN],
    )
    pre = _integer_array(table[PRE_COLUMN].to_numpy(), PRE_COLUMN)
    post = _integer_array(table[POST_COLUMN].to_numpy(), POST_COLUMN)
    strengths = table[WEIGHT_COLUMN].to_numpy().astype(np.float32, copy=False)
    del table
    if len(pre) == 0 or not np.isfinite(strengths).all():
        raise ValueError("Missing or non-finite connectivity")
    if max(int(pre.max()), int(post.max())) >= n_source:
        raise ValueError("Connectivity indices do not match completeness row count")
    n_source_edges = len(pre)
    positive = int((strengths > 0).sum())
    negative = int((strengths < 0).sum())
    zero = int((strengths == 0).sum())
    if source_indices is not None:
        selection = _integer_array(np.asarray(list(source_indices)), "source_indices")
        if len(selection) < 1 or len(np.unique(selection)) != len(selection):
            raise ValueError("source_indices must be nonempty and unique")
        if selection.max() >= n_source:
            raise ValueError("source_indices outside source graph")
        selection_method = "explicit upstream indices (exact checkpoint reconstruction)"
    elif max_neurons is None or max_neurons == n_source:
        selection = np.arange(n_source, dtype=np.int32)
        selection_method = "full upstream graph"
    else:
        selection, seed_index = _connected_selection(
            pre, post, strengths, n_source, max_neurons, seed_index
        )
        selection_method = "strongest-neighbor BFS over weak graph; induced subset"
    remapping = np.full(n_source, -1, dtype=np.int32)
    remapping[selection] = np.arange(len(selection), dtype=np.int32)
    keep = (remapping[pre] >= 0) & (remapping[post] >= 0)
    return ConnectomeGraph(
        node_ids=tuple(ids[int(i)] for i in selection),
        source_indices=selection,
        pre=remapping[pre[keep]],
        post=remapping[post[keep]],
        signed_weights=strengths[keep].copy(),
        provenance={
            "dataset": "FlyWire FAFB materialization v783",
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "verified_source_files": checked,
            "source_neurons": n_source,
            "published_atlas_neurons": PUBLISHED_ATLAS_NEURONS,
            "count_note": (
                "This pinned upstream v783 model file contains 138639 neurons; "
                "the atlas publication reports 139255. No missing rows are padded."
            ),
            "source_directed_edges": n_source_edges,
            "source_positive_edges": positive,
            "source_negative_edges": negative,
            "source_zero_edges": zero,
            "selection_method": selection_method,
            "seed_index": seed_index,
            "anatomical_selection": False,
            "index_convention": "CSV row order, as in upstream model.py; never ID sort",
            "edge_units": "upstream signed connection count",
        },
    )
