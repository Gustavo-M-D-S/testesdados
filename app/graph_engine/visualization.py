"""
Graph Visualization Module
==========================
Renders graphs using multiple backends:

1. Plotly — interactive, web-based (default)
2. PyVis — interactive, HTML-based with physics simulation
3. NetworkX + Matplotlib — static, publication-quality
4. 2D Projection — PCA/UMAP/t-SNE of embeddings with community colors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx

from loguru import logger


@dataclass
class RenderConfig:
    """Configuration for graph rendering."""

    max_nodes_plotly: int = 2000
    max_nodes_pyvis: int = 5000
    node_size: int = 8
    edge_width: float = 0.3
    colors: List[str] = field(default_factory=lambda: [
        "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
        "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ])


class GraphRenderer:
    """
    Renders graphs using multiple backends.

    Usage:
        renderer = GraphRenderer()
        fig = renderer.plotly(G)           # Plotly figure
        html = renderer.pyvis(G)           # PyVis HTML
        fig = renderer.embeddings(X, labels)  # 2D projection
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()

    # ------------------------------------------------------------------
    # Plotly Rendering
    # ------------------------------------------------------------------

    def plotly(
        self,
        G: nx.Graph,
        node_labels: Optional[np.ndarray] = None,
        title: str = "Graph Visualization",
        show_edges: bool = True,
    ) -> Any:
        """
        Render graph using Plotly.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph.
        node_labels : np.ndarray, optional
            Community labels for coloring.
        title : str
            Plot title.
        show_edges : bool
            Whether to draw edges.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        # Subsample if too large
        if G.number_of_nodes() > self.config.max_nodes_plotly:
            logger.warning(f"Graph too large ({G.number_of_nodes()} nodes), subsampling to {self.config.max_nodes_plotly}")
            nodes_sample = list(G.nodes())[:self.config.max_nodes_plotly]
            G = G.subgraph(nodes_sample)

        # Compute layout
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

        # Edge trace
        edge_traces = []
        if show_edges and G.number_of_edges() > 0:
            edge_x = []
            edge_y = []
            edge_colors = []

            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_colors.append(data.get("edge_type", "default"))

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=self.config.edge_width, color="#888"),
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node label
            attrs = G.nodes[node]
            label_parts = [str(node)]
            for key in ["company", "partner", "movement", "id"]:
                if key in attrs:
                    val = str(attrs[key])
                    if len(val) > 30:
                        val = val[:30] + "..."
                    label_parts.append(f"{key}: {val}")
            node_text.append("<br>".join(label_parts))

            # Color by community label
            if node_labels is not None:
                nodes_list = list(G.nodes())
                if node in nodes_list:
                    idx = nodes_list.index(node)
                    if idx < len(node_labels):
                        cid = int(node_labels[idx])
                        node_colors.append(self.config.colors[cid % len(self.config.colors)])
                    else:
                        node_colors.append("#888")
                else:
                    node_colors.append("#888")
            else:
                # Color by degree
                deg = G.degree(node)
                node_colors.append(deg)
                node_sizes.append(self.config.node_size + deg * 0.5)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes if node_sizes else self.config.node_size,
                color=node_colors,
                colorscale="Viridis" if node_labels is None else None,
                showscale=node_labels is None,
                line=dict(width=1, color="white"),
            ),
            text=node_text,
            hoverinfo="text",
            showlegend=False,
        )

        fig = go.Figure(data=[*edge_traces, node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
        )

        return fig

    # ------------------------------------------------------------------
    # PyVis Rendering
    # ------------------------------------------------------------------

    def pyvis(
        self,
        G: nx.Graph,
        node_labels: Optional[np.ndarray] = None,
        output_path: str = "graph.html",
        physics: bool = True,
    ) -> str:
        """
        Render graph using PyVis (interactive HTML with physics).

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph.
        node_labels : np.ndarray, optional
            Community labels for coloring.
        output_path : str
            Path to save HTML file.
        physics : bool
            Enable physics simulation.

        Returns
        -------
        str
            Path to generated HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvis not installed. Install with: pip install pyvis")
            raise

        # Subsample if too large
        if G.number_of_nodes() > self.config.max_nodes_pyvis:
            logger.warning(f"Graph too large, subsampling to {self.config.max_nodes_pyvis}")
            nodes_sample = list(G.nodes())[:self.config.max_nodes_pyvis]
            G = G.subgraph(nodes_sample)

        net = Network(height="800px", width="100%", directed=False)
        net.set_options("""
        {
            "nodes": {
                "font": {"color": "#e0e0e0", "size": 10},
                "borderWidth": 1,
                "borderWidthSelected": 2
            },
            "edges": {
                "color": {"color": "#555", "opacity": 0.3},
                "width": 0.5
            },
            "physics": {"enabled": """ + str(physics).lower() + """,
                         "barnesHut": {"gravitationalConstant": -3000,
                                       "springLength": 100}}
        }
        """)

        nodes_list = list(G.nodes())

        for i, node in enumerate(nodes_list):
            label = str(node)
            if len(label) > 20:
                label = label[:20] + "..."

            # Color
            if node_labels is not None and i < len(node_labels):
                cid = int(node_labels[i])
                color = self.config.colors[cid % len(self.config.colors)]
            else:
                color = "#636efa"

            net.add_node(str(node), label=label, color=color, size=self.config.node_size)

        for u, v in G.edges():
            net.add_edge(str(u), str(v), width=self.config.edge_width)

        net.show(output_path)
        logger.info(f"PyVis graph saved to {output_path}")

        return output_path

    # ------------------------------------------------------------------
    # 2D Embedding Projection
    # ------------------------------------------------------------------

    def embeddings_2d(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "umap",
        title: str = "Node Embeddings (2D Projection)",
    ) -> Any:
        """
        Project node embeddings to 2D and plot with community colors.

        Parameters
        ----------
        X : np.ndarray
            (N, d) embedding matrix.
        labels : np.ndarray, optional
            Community labels for coloring.
        method : str
            'umap', 'pca', or 'tsne'.
        title : str
            Plot title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.express as px

        # Reduce to 2D
        if method == "pca":
            from sklearn.decomposition import PCA
            model = PCA(n_components=2)
            X_2d = model.fit_transform(X)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            model = TSNE(n_components=2, perplexity=min(30, X.shape[0] - 1), random_state=42)
            X_2d = model.fit_transform(X)
        elif method == "umap":
            try:
                import umap
                model = umap.UMAP(n_components=2, random_state=42)
                X_2d = model.fit_transform(X)
            except ImportError:
                logger.warning("umap not installed, falling back to PCA")
                from sklearn.decomposition import PCA
                model = PCA(n_components=2)
                X_2d = model.fit_transform(X)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            "x": X_2d[:, 0],
            "y": X_2d[:, 1],
            "label": labels if labels is not None else ["node"] * len(X),
        })

        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="label" if labels is not None else None,
            title=title,
            opacity=0.7,
            color_discrete_sequence=self.config.colors,
            labels={"label": "Community"},
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )

        return fig

    # ------------------------------------------------------------------
    # Community Size Distribution
    # ------------------------------------------------------------------

    def community_distribution(
        self,
        labels: np.ndarray,
        title: str = "Community Size Distribution",
    ) -> Any:
        """
        Bar chart of community sizes.

        Parameters
        ----------
        labels : np.ndarray
            Community labels.
        title : str
            Plot title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.express as px

        counts = pd.Series(labels).value_counts().sort_index()
        df_plot = pd.DataFrame({
            "Community": counts.index,
            "Size": counts.values,
        })

        fig = px.bar(
            df_plot,
            x="Community",
            y="Size",
            title=title,
            color="Community",
            color_discrete_sequence=self.config.colors,
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#333"),
        )

        return fig

    # ------------------------------------------------------------------
    # Degree Distribution
    # ------------------------------------------------------------------

    def degree_distribution(
        self,
        G: nx.Graph,
        title: str = "Degree Distribution",
    ) -> Any:
        """
        Histogram of node degrees.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph.
        title : str
            Plot title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.express as px

        degrees = [d for _, d in G.degree()]
        df_plot = pd.DataFrame({"Degree": degrees})

        fig = px.histogram(
            df_plot,
            x="Degree",
            title=title,
            nbins=50,
            color_discrete_sequence=["#636efa"],
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#333"),
            bargap=0.05,
        )

        return fig