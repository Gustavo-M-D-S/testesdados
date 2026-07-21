from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

import hdbscan
import community as community_louvain
import igraph as ig
import leidenalg

from loguru import logger


# -------------------------
# RESULT
# -------------------------

@dataclass
class ClusteringResult:
    labels: np.ndarray
    method: str
    score: float


# -------------------------
# CLUSTERING ENGINE
# -------------------------

class CommunityDetector:
    """
    Community detection for graph embeddings.
    """

    def __init__(self):
        pass

    # -------------------------
    # METRICS
    # -------------------------

    def _evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Internal clustering quality metric.
        """
        if len(set(labels)) <= 1:
            return -1.0

        return silhouette_score(X, labels)

    # -------------------------
    # KMEANS
    # -------------------------

    def kmeans(self, X: np.ndarray, k: int = 8) -> ClusteringResult:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)

        score = self._evaluate(X, labels)

        return ClusteringResult(labels, "kmeans", score)

    # -------------------------
    # HDBSCAN
    # -------------------------

    def hdbscan(self, X: np.ndarray) -> ClusteringResult:
        model = hdbscan.HDBSCAN(min_cluster_size=15)
        labels = model.fit_predict(X)

        score = self._evaluate(X, labels)

        return ClusteringResult(labels, "hdbscan", score)

    # -------------------------
    # SPECTRAL
    # -------------------------

    def spectral(self, X: np.ndarray, k: int = 8) -> ClusteringResult:
        model = SpectralClustering(
            n_clusters=k,
            assign_labels="kmeans",
            random_state=42,
        )

        labels = model.fit_predict(X)
        score = self._evaluate(X, labels)

        return ClusteringResult(labels, "spectral", score)

    # -------------------------
    # LOUVAIN (GRAPH-BASED)
    # -------------------------

    def louvain(self, edge_index: np.ndarray) -> ClusteringResult:
        G = ig.Graph()

        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_vertices(int(edge_index.max()) + 1)
        G.add_edges(edges)

        partition = community_louvain.best_partition(G.to_networkx())

        labels = np.array(list(partition.values()))

        score = -1.0  # modularity could be added if needed

        return ClusteringResult(labels, "louvain", score)

    # -------------------------
    # LEIDEN (BEST PRACTICE)
    # -------------------------

    def leiden(self, edge_index: np.ndarray) -> ClusteringResult:
        G = ig.Graph()

        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_vertices(int(edge_index.max()) + 1)
        G.add_edges(edges)

        partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)

        labels = np.zeros(G.vcount())

        for cid, cluster in enumerate(partition):
            for node in cluster:
                labels[node] = cid

        score = partition.modularity

        return ClusteringResult(labels, "leiden", score)

    # -------------------------
    # AUTO SELECT
    # -------------------------

    def auto_cluster(
        self,
        X: np.ndarray,
        edge_index: np.ndarray | None = None,
    ) -> ClusteringResult:

        logger.info("Running clustering ensemble...")

        results = []

        results.append(self.kmeans(X))
        results.append(self.hdbscan(X))
        results.append(self.spectral(X))

        if edge_index is not None:
            results.append(self.louvain(edge_index))
            results.append(self.leiden(edge_index))

        best = max(results, key=lambda r: r.score)

        logger.info(f"Best clustering: {best.method} | score={best.score:.4f}")

        return best