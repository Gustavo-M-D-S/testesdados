"""
Community Detection Module
==========================
Detects communities in the graph using multiple algorithms:

1. Leiden — modularity optimization (best quality)
2. Louvain — modularity optimization (fast)
3. Label Propagation — linear time, scalable
4. Infomap — information-theoretic
5. Spectral Clustering — eigenvector-based
6. KMeans on embeddings — feature-based clustering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from loguru import logger


@dataclass
class CommunityResult:
    """Result of community detection."""

    labels: np.ndarray  # (N,) community label for each node
    method: str
    score: float  # modularity or silhouette score
    n_communities: int
    node_order: List[str]  # node IDs in the same order as labels
    metadata: Dict = field(default_factory=dict)


class CommunityDetector:
    """
    Detects communities in a NetworkX graph using multiple algorithms.

    Usage:
        detector = CommunityDetector()
        result = detector.leiden(G)
        result = detector.louvain(G)
        result = detector.label_propagation(G)
        result = detector.spectral(G, k=8)
        result = detector.all_methods(G)  # runs everything
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Leiden Algorithm
    # ------------------------------------------------------------------

    def leiden(self, G: nx.Graph, resolution: float = 1.0) -> CommunityResult:
        """
        Leiden community detection via igraph.

        Guarantees connected communities. Best modularity optimization.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            logger.error("leidenalg/igraph not installed. Install with: pip install leidenalg igraph")
            raise

        logger.info("Running Leiden community detection...")

        # Convert NetworkX to iGraph
        ig_G = ig.Graph.from_networkx(G)

        # Run Leiden
        # Check if graph has weight attribute
        has_weights = any("weight" in d for _, _, d in G.edges(data=True))
        weights = "weight" if has_weights else None

        partition = leidenalg.find_partition(
            ig_G,
            leidenalg.ModularityVertexPartition,
            weights=weights,
        )

        # Extract labels
        labels = np.zeros(ig_G.vcount(), dtype=np.int64)
        for cid, cluster in enumerate(partition):
            for node in cluster:
                labels[node] = cid

        n_communities = len(partition)
        modularity = partition.modularity

        logger.info(f"Leiden: {n_communities} communities, modularity = {modularity:.4f}")

        return CommunityResult(
            labels=labels,
            method="leiden",
            score=modularity,
            n_communities=n_communities,
            node_order=list(G.nodes()),
            metadata={"resolution": resolution},
        )

    # ------------------------------------------------------------------
    # Louvain Algorithm
    # ------------------------------------------------------------------

    def louvain(self, G: nx.Graph, resolution: float = 1.0) -> CommunityResult:
        """
        Louvain community detection via python-louvain (community package).

        Fast modularity optimization. Well-suited for large graphs.
        """
        try:
            import community as community_louvain
        except ImportError:
            logger.error("python-louvain not installed. Install with: pip install python-louvain")
            raise

        logger.info("Running Louvain community detection...")

        partition_dict = community_louvain.best_partition(G, resolution=resolution)

        # Convert dict to array in node order
        nodes = list(G.nodes())
        labels = np.array([partition_dict[n] for n in nodes], dtype=np.int64)

        n_communities = len(set(partition_dict.values()))
        modularity = community_louvain.modularity(partition_dict, G)

        logger.info(f"Louvain: {n_communities} communities, modularity = {modularity:.4f}")

        return CommunityResult(
            labels=labels,
            method="louvain",
            score=modularity,
            n_communities=n_communities,
            node_order=nodes,
            metadata={"resolution": resolution},
        )

    # ------------------------------------------------------------------
    # Label Propagation
    # ------------------------------------------------------------------

    def label_propagation(self, G: nx.Graph) -> CommunityResult:
        """
        Label Propagation Algorithm (LPA).

        Each node adopts the most frequent label among its neighbors.
        O(m) time complexity — very fast for large graphs.
        """
        from networkx.algorithms.community import label_propagation_communities

        logger.info("Running Label Propagation...")

        communities = list(label_propagation_communities(G))

        # Build label array
        nodes = list(G.nodes())
        labels = np.zeros(len(nodes), dtype=np.int64)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        for cid, community in enumerate(communities):
            for node in community:
                labels[node_to_idx[node]] = cid

        n_communities = len(communities)

        # Compute modularity for comparison
        try:
            import community as community_louvain
            partition_dict = {n: labels[i] for i, n in enumerate(nodes)}
            modularity = community_louvain.modularity(partition_dict, G)
        except Exception:
            modularity = -1.0

        logger.info(f"Label Propagation: {n_communities} communities, modularity = {modularity:.4f}")

        return CommunityResult(
            labels=labels,
            method="label_propagation",
            score=modularity,
            n_communities=n_communities,
            node_order=nodes,
        )

    # ------------------------------------------------------------------
    # Infomap
    # ------------------------------------------------------------------

    def infomap(self, G: nx.Graph) -> CommunityResult:
        """
        Infomap community detection.

        Information-theoretic approach minimizing the map equation.
        Captures flow patterns in the graph.
        """
        try:
            import igraph as ig
        except ImportError:
            logger.error("igraph not installed")
            raise

        logger.info("Running Infomap...")

        ig_G = ig.Graph.from_networkx(G)
        communities = ig_G.community_infomap()

        nodes = list(G.nodes())
        labels = np.array(communities.membership, dtype=np.int64)
        n_communities = len(set(labels))

        # Modularity for comparison
        try:
            import community as community_louvain
            partition_dict = {n: labels[i] for i, n in enumerate(nodes)}
            modularity = community_louvain.modularity(partition_dict, G)
        except Exception:
            modularity = -1.0

        logger.info(f"Infomap: {n_communities} communities, modularity = {modularity:.4f}")

        return CommunityResult(
            labels=labels,
            method="infomap",
            score=modularity,
            n_communities=n_communities,
            node_order=nodes,
        )

    # ------------------------------------------------------------------
    # Spectral Clustering
    # ------------------------------------------------------------------

    def spectral(self, G: nx.Graph, k: int = 8) -> CommunityResult:
        """
        Spectral clustering on the graph Laplacian.

        Uses the eigenvectors of L = D - A to embed nodes,
        then applies k-means.
        """
        from sklearn.cluster import SpectralClustering as SkSpectral
        from sklearn.metrics import silhouette_score

        logger.info(f"Running Spectral Clustering (k={k})...")

        # Get adjacency matrix
        A = nx.adjacency_matrix(G).astype(np.float64)
        n = A.shape[0]
        k = min(k, n - 1)

        if k < 2:
            return CommunityResult(
                labels=np.zeros(n, dtype=np.int64),
                method="spectral",
                score=-1.0,
                n_communities=1,
                node_order=list(G.nodes()),
            )

        model = SkSpectral(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )

        labels = model.fit_predict(A)

        # Silhouette score
        if len(set(labels)) > 1:
            sil = silhouette_score(A, labels, metric="precomputed")
        else:
            sil = -1.0

        n_communities = len(set(labels))

        logger.info(f"Spectral: {n_communities} communities, silhouette = {sil:.4f}")

        return CommunityResult(
            labels=labels,
            method="spectral",
            score=sil,
            n_communities=n_communities,
            node_order=list(G.nodes()),
            metadata={"k": k},
        )

    # ------------------------------------------------------------------
    # KMeans on Node Features
    # ------------------------------------------------------------------

    def kmeans_features(
        self,
        G: nx.Graph,
        k: int = 8,
    ) -> CommunityResult:
        """
        KMeans clustering on node feature vectors.

        Uses the 'features' attribute stored on each node.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        logger.info(f"Running KMeans on node features (k={k})...")

        nodes = list(G.nodes())
        n = len(nodes)
        k = min(k, n - 1)

        # Extract feature matrix
        X_list = []
        for node in nodes:
            feats = G.nodes[node].get("features")
            if feats is not None:
                X_list.append(feats)

        if not X_list:
            logger.warning("No features found on nodes. Using degree features.")
            X_list = [[G.degree(n)] for n in nodes]

        X = np.array(X_list)

        if k < 2:
            return CommunityResult(
                labels=np.zeros(n, dtype=np.int64),
                method="kmeans_features",
                score=-1.0,
                n_communities=1,
                node_order=nodes,
            )

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = -1.0

        n_communities = len(set(labels))

        logger.info(f"KMeans (features): {n_communities} clusters, silhouette = {sil:.4f}")

        return CommunityResult(
            labels=labels,
            method="kmeans_features",
            score=sil,
            n_communities=n_communities,
            node_order=nodes,
            metadata={"k": k},
        )

    # ------------------------------------------------------------------
    # Run All Methods
    # ------------------------------------------------------------------

    def all_methods(
        self,
        G: nx.Graph,
        k_spectral: int = 8,
        k_kmeans: int = 8,
    ) -> Dict[str, CommunityResult]:
        """
        Run all available community detection methods.

        Returns a dict mapping method_name -> CommunityResult.
        """
        results: Dict[str, CommunityResult] = {}

        methods = [
            ("leiden", lambda: self.leiden(G)),
            ("louvain", lambda: self.louvain(G)),
            ("label_propagation", lambda: self.label_propagation(G)),
            ("spectral", lambda: self.spectral(G, k=k_spectral)),
            ("kmeans_features", lambda: self.kmeans_features(G, k=k_kmeans)),
        ]

        # Infomap requires igraph
        try:
            import igraph
            methods.append(("infomap", lambda: self.infomap(G)))
        except ImportError:
            pass

        for name, method_fn in methods:
            try:
                result = method_fn()
                results[name] = result
                logger.info(f"{name}: {result.n_communities} communities, score = {result.score:.4f}")
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        return results