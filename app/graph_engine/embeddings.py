"""
Node Embeddings Module
======================
Generates node embeddings from the graph using multiple techniques:

1. Node2Vec — biased random walk + Word2Vec (via gensim)
2. DeepWalk — uniform random walk + Word2Vec
3. GraphSAGE — inductive neighborhood aggregation (via PyG)
4. GCN — graph convolutional network (via PyG)
5. Laplacian Eigenmaps — spectral embedding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from loguru import logger


@dataclass
class EmbeddingResult:
    """Result of node embedding generation."""

    embeddings: np.ndarray  # (N, d) embedding matrix
    method: str
    dimension: int
    node_order: List[str]  # node IDs in the same order as embeddings
    metadata: Dict = field(default_factory=dict)


class NodeEmbedder:
    """
    Generates node embeddings from a NetworkX graph.

    Usage:
        embedder = NodeEmbedder()
        result = embedder.node2vec(G, dimensions=64)
        result = embedder.deepwalk(G, dimensions=64)
        result = embedder.laplacian(G, dimensions=64)
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Node2Vec
    # ------------------------------------------------------------------

    def node2vec(
        self,
        G: nx.Graph,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 10,
        p: float = 1.0,  # return parameter
        q: float = 1.0,  # in-out parameter
        workers: int = 4,
    ) -> EmbeddingResult:
        """
        Node2Vec embeddings via biased random walks + Word2Vec.

        Parameters
        ----------
        p : float
            Return parameter. Higher p = more BFS-like walks.
        q : float
            In-out parameter. Higher q = more DFS-like walks.
        """
        try:
            from node2vec import Node2Vec
        except ImportError:
            logger.error("node2vec not installed. Install with: pip install node2vec")
            raise

        logger.info(f"Running Node2Vec (dim={dimensions}, walks={num_walks}, len={walk_length})...")

        # Node2Vec expects a connected graph; warn if disconnected
        if not nx.is_connected(G):
            logger.warning("Graph is disconnected. Node2Vec will run on the largest component.")

        n2v = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            quiet=True,
        )

        model = n2v.fit(window=10, min_count=1, batch_words=4)

        # Extract embeddings in node order
        nodes = list(G.nodes())
        embeddings = np.array([model.wv[str(n)] for n in nodes])

        logger.info(f"Node2Vec embeddings: {embeddings.shape}")

        return EmbeddingResult(
            embeddings=embeddings,
            method="node2vec",
            dimension=dimensions,
            node_order=nodes,
            metadata={"p": p, "q": q, "walk_length": walk_length, "num_walks": num_walks},
        )

    # ------------------------------------------------------------------
    # DeepWalk
    # ------------------------------------------------------------------

    def deepwalk(
        self,
        G: nx.Graph,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 10,
        workers: int = 4,
    ) -> EmbeddingResult:
        """
        DeepWalk embeddings via uniform random walks + Word2Vec.

        Equivalent to Node2Vec with p = q = 1.0.
        """
        return self.node2vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=1.0,
            q=1.0,
            workers=workers,
        )

    # ------------------------------------------------------------------
    # Laplacian Eigenmaps
    # ------------------------------------------------------------------

    def laplacian(
        self,
        G: nx.Graph,
        dimensions: int = 64,
    ) -> EmbeddingResult:
        """
        Laplacian Eigenmaps — spectral embedding using the graph Laplacian.

        Uses the eigenvectors corresponding to the smallest non-zero
        eigenvalues of the normalized Laplacian.
        """
        from sklearn.decomposition import PCA

        logger.info(f"Running Laplacian Eigenmaps (dim={dimensions})...")

        # Compute normalized Laplacian
        L = nx.normalized_laplacian_matrix(G).astype(np.float64)

        # Compute eigenvalues and eigenvectors
        # Use scipy.sparse.linalg.eigsh for efficiency
        from scipy.sparse.linalg import eigsh

        n = L.shape[0]
        k = min(dimensions + 1, n - 1)  # +1 because we skip the first (smallest)

        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip the first eigenvector (constant, eigenvalue ≈ 0)
        embeddings = eigenvectors[:, 1 : k - 1] if k > 2 else eigenvectors[:, :k]

        # If we got fewer dimensions than requested, pad with PCA
        if embeddings.shape[1] < dimensions:
            pca = PCA(n_components=dimensions)
            embeddings = pca.fit_transform(embeddings)

        nodes = list(G.nodes())

        logger.info(f"Laplacian Eigenmaps: {embeddings.shape}")

        return EmbeddingResult(
            embeddings=embeddings,
            method="laplacian_eigenmaps",
            dimension=embeddings.shape[1],
            node_order=nodes,
        )

    # ------------------------------------------------------------------
    # GraphSAGE (via PyTorch Geometric)
    # ------------------------------------------------------------------

    def graphsage(
        self,
        G: nx.Graph,
        dimensions: int = 64,
        hidden_channels: int = 128,
        epochs: int = 50,
    ) -> EmbeddingResult:
        """
        GraphSAGE embeddings via PyTorch Geometric.

        Converts NetworkX graph to PyG Data, trains a GraphSAGE model
        in an unsupervised manner, and extracts node embeddings.
        """
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import SAGEConv
            from torch_geometric.data import Data
            from torch_geometric.utils import from_networkx
        except ImportError:
            logger.error("PyTorch Geometric not installed")
            raise

        logger.info(f"Running GraphSAGE (dim={dimensions}, epochs={epochs})...")

        # Convert NetworkX to PyG
        data = from_networkx(G)
        data.x = torch.randn(data.num_nodes, hidden_channels)  # random input features

        # Define simple GraphSAGE model
        class SAGE(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SAGE(hidden_channels, hidden_channels, dimensions).to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop (simple reconstruction loss)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # Simple loss: nodes connected should have similar embeddings
            loss = F.mse_loss(
                out[data.edge_index[0]],
                out[data.edge_index[1]],
            )
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.debug(f"GraphSAGE epoch {epoch+1}/{epochs}, loss = {loss.item():.4f}")

        # Extract embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu().numpy()

        nodes = list(G.nodes())

        logger.info(f"GraphSAGE embeddings: {embeddings.shape}")

        return EmbeddingResult(
            embeddings=embeddings,
            method="graphsage",
            dimension=dimensions,
            node_order=nodes,
        )

    # ------------------------------------------------------------------
    # GCN (via PyTorch Geometric)
    # ------------------------------------------------------------------

    def gcn(
        self,
        G: nx.Graph,
        dimensions: int = 64,
        hidden_channels: int = 128,
        epochs: int = 50,
    ) -> EmbeddingResult:
        """
        GCN embeddings via PyTorch Geometric.

        Similar to GraphSAGE but uses GCN convolution.
        """
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv
            from torch_geometric.data import Data
            from torch_geometric.utils import from_networkx
        except ImportError:
            logger.error("PyTorch Geometric not installed")
            raise

        logger.info(f"Running GCN (dim={dimensions}, epochs={epochs})...")

        data = from_networkx(G)
        data.x = torch.randn(data.num_nodes, hidden_channels)

        class GCN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GCN(hidden_channels, hidden_channels, dimensions).to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(
                out[data.edge_index[0]],
                out[data.edge_index[1]],
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu().numpy()

        nodes = list(G.nodes())

        logger.info(f"GCN embeddings: {embeddings.shape}")

        return EmbeddingResult(
            embeddings=embeddings,
            method="gcn",
            dimension=dimensions,
            node_order=nodes,
        )

    # ------------------------------------------------------------------
    # All Methods
    # ------------------------------------------------------------------

    def all_methods(
        self,
        G: nx.Graph,
        dimensions: int = 64,
    ) -> Dict[str, EmbeddingResult]:
        """
        Run all available embedding methods.

        Returns a dict mapping method_name -> EmbeddingResult.
        """
        results: Dict[str, EmbeddingResult] = {}

        methods = [
            ("laplacian_eigenmaps", lambda: self.laplacian(G, dimensions=dimensions)),
        ]

        # Node2Vec
        try:
            import node2vec
            methods.append(("node2vec", lambda: self.node2vec(G, dimensions=dimensions)))
        except ImportError:
            pass

        # GraphSAGE / GCN
        try:
            import torch_geometric
            methods.append(("graphsage", lambda: self.graphsage(G, dimensions=dimensions)))
            methods.append(("gcn", lambda: self.gcn(G, dimensions=dimensions)))
        except ImportError:
            pass

        for name, method_fn in methods:
            try:
                result = method_fn()
                results[name] = result
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        return results