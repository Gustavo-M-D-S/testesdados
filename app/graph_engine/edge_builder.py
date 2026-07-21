"""
Edge Building Module
====================
Constructs edges between nodes using multiple strategies:

1. Cosine Similarity kNN — connects each node to its k most similar neighbors
2. ε-Neighborhood — connects nodes whose similarity exceeds a threshold
3. Shared Attribute — connects nodes sharing the same categorical value
4. Temporal Proximity — connects entries within a time window
5. Heterogeneous — combines multiple strategies as different edge types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import networkx as nx

from loguru import logger


@dataclass
class EdgeConfig:
    """Configuration for edge construction."""

    # kNN strategy
    knn_k: int = 15
    knn_metric: str = "cosine"  # 'cosine' or 'euclidean'

    # ε-neighborhood strategy
    epsilon_threshold: Optional[float] = None  # if None, uses percentile
    epsilon_percentile: float = 95.0  # percentile for auto-threshold

    # Shared attribute strategy
    shared_attributes: List[str] = field(default_factory=lambda: [])

    # Temporal proximity
    temporal_window_days: int = 7
    temporal_col: Optional[str] = None

    # Scaling
    max_edges_per_node: int = 50  # maximum neighbors per node (any strategy)

    # Approximate nearest neighbors (for N > 100k)
    use_annoy: bool = False
    annoy_n_trees: int = 50


@dataclass
class EdgeResult:
    """Result of edge construction."""

    edges: np.ndarray  # (2, E) array of edges
    edge_weights: np.ndarray  # (E,) array of edge weights
    strategy: str
    n_edges: int
    metadata: Dict = field(default_factory=dict)


class EdgeBuilder:
    """
    Builds edges between nodes using configurable strategies.

    The builder takes a feature matrix X and produces edge lists
    that can be used to construct a NetworkX or PyG graph.
    """

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()

    # ------------------------------------------------------------------
    # Cosine Similarity kNN
    # ------------------------------------------------------------------

    def _cosine_similarity(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms
        return X_norm @ X_norm.T

    def _euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix."""
        # Use squared Euclidean for efficiency
        sum_X = np.sum(X ** 2, axis=1)
        dist_sq = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
        dist_sq = np.clip(dist_sq, 0, None)  # numerical stability
        return np.sqrt(dist_sq)

    def knn_edges(self, X: np.ndarray) -> EdgeResult:
        """
        Build kNN graph based on cosine similarity.

        Uses sklearn NearestNeighbors with ball_tree/brute algorithm.
        For large datasets, this avoids materializing the full similarity matrix.
        """
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        k = min(self.config.knn_k, n - 1)

        logger.info(f"Computing {k}-NN graph on {n} nodes...")

        metric = "cosine" if self.config.knn_metric == "cosine" else "euclidean"

        # For large n, use brute force with auto algorithm
        # For n < 100k, ball_tree is efficient; for larger, brute is better
        algorithm = "ball_tree" if n < 50000 else "brute"

        nn = NearestNeighbors(
            n_neighbors=k + 1,  # +1 because self is included
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1,
        )
        nn.fit(X)

        # Query k+1 neighbors (first is self with distance ~0)
        distances, indices = nn.kneighbors(X)

        # Build edge list (skip self)
        edges = []
        weights = []

        for i in range(n):
            for j_idx in range(1, k + 1):  # skip index 0 (self)
                j = indices[i, j_idx]
                d = distances[i, j_idx]
                edges.append((i, j))
                # Convert distance to similarity weight
                if metric == "cosine":
                    w = 1.0 - d  # cosine distance = 1 - cosine similarity
                else:
                    w = np.exp(-d)
                weights.append(max(0.0, w))

        edges_arr = np.array(edges, dtype=np.int64).T
        weights_arr = np.array(weights, dtype=np.float64)

        logger.info(f"kNN edges: {edges_arr.shape[1]} edges")

        return EdgeResult(
            edges=edges_arr,
            edge_weights=weights_arr,
            strategy=f"knn_k={k}",
            n_edges=edges_arr.shape[1],
        )

    def _annoy_knn(self, X: np.ndarray) -> EdgeResult:
        """Use Annoy for approximate kNN on large datasets."""
        try:
            from annoy import AnnoyIndex
        except ImportError:
            logger.warning("Annoy not installed, falling back to sklearn")
            return self._sklearn_knn(X)

        n, d = X.shape
        k = min(self.config.knn_k, n - 1)

        logger.info(f"Building Annoy index ({n} nodes, {d} dims)...")

        # Normalize for cosine
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms

        index = AnnoyIndex(d, "angular")
        for i in range(n):
            index.add_item(i, X_norm[i].tolist())
        index.build(self.config.annoy_n_trees)

        edges = []
        weights = []

        for i in range(n):
            neighbors = index.get_nns_by_item(i, k + 1)  # +1 includes self
            for j in neighbors:
                if i != j:
                    edges.append((i, j))
                    # Angular distance → cosine similarity
                    w = 1.0 - index.get_distance(i, j)
                    weights.append(max(0.0, w))

        edges_arr = np.array(edges, dtype=np.int64).T
        weights_arr = np.array(weights, dtype=np.float64)

        logger.info(f"Annoy kNN edges: {edges_arr.shape[1]}")

        return EdgeResult(
            edges=edges_arr,
            edge_weights=weights_arr,
            strategy=f"annoy_knn_k={k}",
            n_edges=edges_arr.shape[1],
        )

    def _sklearn_knn(self, X: np.ndarray) -> EdgeResult:
        """Fallback using sklearn's NearestNeighbors."""
        from sklearn.neighbors import NearestNeighbors

        n = X.shape[0]
        k = min(self.config.knn_k, n - 1)

        metric = "cosine" if self.config.knn_metric == "cosine" else "euclidean"
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
        nn.fit(X)

        distances, indices = nn.kneighbors(X)

        edges = []
        weights = []

        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                if i != j:
                    edges.append((i, j))
                    w = np.exp(-distances[i, j_idx])
                    weights.append(max(0.0, w))

        edges_arr = np.array(edges, dtype=np.int64).T
        weights_arr = np.array(weights, dtype=np.float64)

        return EdgeResult(
            edges=edges_arr,
            edge_weights=weights_arr,
            strategy=f"sklearn_knn_k={k}",
            n_edges=edges_arr.shape[1],
        )

    # ------------------------------------------------------------------
    # ε-Neighborhood
    # ------------------------------------------------------------------

    def epsilon_edges(self, X: np.ndarray) -> EdgeResult:
        """
        Build ε-neighborhood graph.

        Connects nodes whose similarity exceeds a threshold.
        Threshold is either given or set as a percentile of all similarities.
        """
        n = X.shape[0]
        if n > 50000:
            # Sample to determine threshold
            logger.info("Large dataset: sampling to determine ε threshold")

        # Compute similarity
        sim = self._cosine_similarity(X)

        # Determine threshold
        if self.config.epsilon_threshold is not None:
            tau = self.config.epsilon_threshold
        else:
            # Use percentile of upper triangle (excluding diagonal)
            triu = sim[np.triu_indices(n, k=1)]
            tau = np.percentile(triu, self.config.epsilon_percentile)

        logger.info(f"ε threshold: {tau:.4f} (percentile {self.config.epsilon_percentile})")

        # Find edges above threshold
        rows, cols = np.where(sim > tau)
        # Remove self-loops and duplicates (keep only upper triangle)
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]

        edges = np.stack([rows, cols], axis=0)
        weights = sim[rows, cols]

        logger.info(f"ε-neighborhood edges: {edges.shape[1]}")

        return EdgeResult(
            edges=edges,
            edge_weights=weights,
            strategy=f"epsilon_tau={tau:.4f}",
            n_edges=edges.shape[1],
        )

    # ------------------------------------------------------------------
    # Shared Attribute Edges
    # ------------------------------------------------------------------

    def shared_attribute_edges(
        self,
        df: pd.DataFrame,
        attribute_cols: Optional[List[str]] = None,
    ) -> EdgeResult:
        """
        Build edges connecting nodes that share the same categorical value.

        For each attribute column, creates edges between all pairs of rows
        that have the same value in that column.
        """
        cols = attribute_cols or self.config.shared_attributes
        if not cols:
            return EdgeResult(
                edges=np.empty((2, 0), dtype=np.int64),
                edge_weights=np.empty(0),
                strategy="shared_attr_none",
                n_edges=0,
            )

        edges = []
        weights = []

        for col in cols:
            if col not in df.columns:
                continue

            logger.info(f"Building shared-attribute edges for '{col}'...")

            # Group by column value
            groups = df.groupby(col, sort=False).indices

            for val, indices in groups.items():
                idx = indices.values
                if len(idx) < 2:
                    continue

                # Connect all pairs within this group
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        edges.append((idx[i], idx[j]))
                        weights.append(1.0)

        if not edges:
            return EdgeResult(
                edges=np.empty((2, 0), dtype=np.int64),
                edge_weights=np.empty(0),
                strategy="shared_attr",
                n_edges=0,
            )

        edges_arr = np.array(edges, dtype=np.int64).T
        weights_arr = np.array(weights, dtype=np.float64)

        logger.info(f"Shared attribute edges: {edges_arr.shape[1]}")

        return EdgeResult(
            edges=edges_arr,
            edge_weights=weights_arr,
            strategy=f"shared_attr_{len(cols)}_cols",
            n_edges=edges_arr.shape[1],
        )

    # ------------------------------------------------------------------
    # Heterogeneous Multi-Edge
    # ------------------------------------------------------------------

    def heterogeneous_edges(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
    ) -> Dict[str, EdgeResult]:
        """
        Build multiple edge types for a heterogeneous graph.

        Returns a dict mapping edge_type_name -> EdgeResult.
        """
        results: Dict[str, EdgeResult] = {}

        # Type 1: kNN similarity edges
        knn_res = self.knn_edges(X)
        if knn_res.n_edges > 0:
            results["similarity"] = knn_res

        # Type 2: Shared attribute edges
        if self.config.shared_attributes:
            attr_res = self.shared_attribute_edges(df)
            if attr_res.n_edges > 0:
                results["shared_attribute"] = attr_res

        # Type 3: Temporal proximity edges
        if self.config.temporal_col and self.config.temporal_col in df.columns:
            temp_res = self._temporal_edges(df)
            if temp_res.n_edges > 0:
                results["temporal"] = temp_res

        logger.info(f"Heterogeneous graph with {len(results)} edge types")

        return results

    def _temporal_edges(self, df: pd.DataFrame) -> EdgeResult:
        """
        Connect entries within a temporal window.
        """
        col = self.config.temporal_col
        if col not in df.columns:
            return EdgeResult(
                edges=np.empty((2, 0), dtype=np.int64),
                edge_weights=np.empty(0),
                strategy="temporal",
                n_edges=0,
            )

        try:
            dates = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            return EdgeResult(
                edges=np.empty((2, 0), dtype=np.int64),
                edge_weights=np.empty(0),
                strategy="temporal",
                n_edges=0,
            )

        window = pd.Timedelta(days=self.config.temporal_window_days)
        n = len(df)
        edges = []
        weights = []

        # Sort by date for efficient window scanning
        sorted_idx = np.argsort(dates.values)

        for i in range(n):
            current_date = dates.iloc[sorted_idx[i]]
            if pd.isna(current_date):
                continue

            # Look forward within window
            for j in range(i + 1, n):
                other_date = dates.iloc[sorted_idx[j]]
                if pd.isna(other_date):
                    continue

                diff = abs((other_date - current_date).total_seconds())
                if diff > window.total_seconds():
                    break

                edges.append((sorted_idx[i], sorted_idx[j]))
                # Weight decays with time difference
                w = 1.0 - diff / window.total_seconds()
                weights.append(max(0.0, w))

        if not edges:
            return EdgeResult(
                edges=np.empty((2, 0), dtype=np.int64),
                edge_weights=np.empty(0),
                strategy="temporal",
                n_edges=0,
            )

        edges_arr = np.array(edges, dtype=np.int64).T
        weights_arr = np.array(weights, dtype=np.float64)

        return EdgeResult(
            edges=edges_arr,
            edge_weights=weights_arr,
            strategy=f"temporal_window={self.config.temporal_window_days}d",
            n_edges=edges_arr.shape[1],
        )