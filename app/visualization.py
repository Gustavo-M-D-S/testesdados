from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

import networkx as nx

from torch_geometric.data import HeteroData

from loguru import logger


# -------------------------
# RESULT
# -------------------------

@dataclass
class ProjectionResult:
    embeddings_2d: np.ndarray
    method: str


# -------------------------
# VISUALIZATION ENGINE
# -------------------------

class GraphVisualizer:
    """
    Visualization utilities for embeddings, clusters and graph structure.
    """

    # -------------------------
    # DIMENSION REDUCTION
    # -------------------------

    def reduce(self, X: np.ndarray, method: str = "umap") -> ProjectionResult:
        """
        Reduce embeddings to 2D space.
        """

        logger.info(f"Reducing embeddings using {method}")

        if method == "pca":
            model = PCA(n_components=2)
            X_2d = model.fit_transform(X)

        elif method == "tsne":
            model = TSNE(n_components=2, perplexity=30, random_state=42)
            X_2d = model.fit_transform(X)

        elif method == "umap":
            model = umap.UMAP(n_components=2, random_state=42)
            X_2d = model.fit_transform(X)

        else:
            raise ValueError(f"Unknown method: {method}")

        return ProjectionResult(X_2d, method)

    # -------------------------
    # EMBEDDINGS PLOT
    # -------------------------

    def plot_embeddings(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "umap",
        title: str = "Graph Embeddings",
    ):

        projection = self.reduce(X, method)

        fig = px.scatter(
            x=projection.embeddings_2d[:, 0],
            y=projection.embeddings_2d[:, 1],
            color=labels if labels is not None else None,
            title=title,
            opacity=0.7,
        )

        return fig

    # -------------------------
    # ANOMALY VISUALIZATION
    # -------------------------

    def plot_anomalies(
        self,
        X: np.ndarray,
        anomaly_labels: np.ndarray,
        method: str = "umap",
    ):

        projection = self.reduce(X, method)

        colors = ["green" if a == 0 else "red" for a in anomaly_labels]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=projection.embeddings_2d[:, 0],
                y=projection.embeddings_2d[:, 1],
                mode="markers",
                marker=dict(color=colors, size=6),
            )
        )

        fig.update_layout(title="Anomaly Detection Visualization")

        return fig

    # -------------------------
    # GRAPH VISUALIZATION
    # -------------------------

    def plot_graph(self, data: HeteroData, max_nodes: int = 500):

        logger.info("Building NetworkX graph for visualization")

        G = nx.Graph()

        for edge_type, edge_index in data.edge_index_dict.items():

            src_type, _, dst_type = edge_type

            edge_index = edge_index[:, :max_nodes]

            for src, dst in edge_index.t().tolist():

                G.add_edge(
                    f"{src_type}_{src}",
                    f"{dst_type}_{dst}",
                )

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=6, color="blue"),
            text=list(G.nodes()),
            hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title="Graph Structure",
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
        )

        return fig