"""
Graph Factory Module
====================
Orchestrates the complete graph construction pipeline:

1. Feature extraction from raw DataFrame
2. Edge construction using selected strategies
3. NetworkX graph assembly
4. PyTorch Geometric conversion (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

from loguru import logger

from app.graph_engine.node_features import FeatureExtractor, FeatureConfig, FeatureResult
from app.graph_engine.edge_builder import EdgeBuilder, EdgeConfig, EdgeResult


@dataclass
class GraphFactoryConfig:
    """Configuration for the complete graph factory."""

    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    edge_config: EdgeConfig = field(default_factory=EdgeConfig)
    id_column: Optional[str] = None
    edge_strategy: str = "knn"  # 'knn', 'epsilon', 'shared_attr', 'heterogeneous'
    make_undirected: bool = True
    remove_self_loops: bool = True


@dataclass
class GraphResult:
    """Complete result of graph construction."""

    graph: nx.Graph
    features: FeatureResult
    edges: Dict[str, EdgeResult]
    n_nodes: int
    n_edges: int
    n_edge_types: int
    node_id_to_idx: Dict[str, int]
    idx_to_node_id: Dict[int, str]


class GraphFactory:
    """
    Orchestrates the complete graph construction pipeline.

    Usage:
        factory = GraphFactory()
        result = factory.build(df)
        G = result.graph  # NetworkX graph
    """

    def __init__(self, config: Optional[GraphFactoryConfig] = None):
        self.config = config or GraphFactoryConfig()
        self.feature_extractor = FeatureExtractor(self.config.feature_config)
        self.edge_builder = EdgeBuilder(self.config.edge_config)

    # ------------------------------------------------------------------
    # Main Build Pipeline
    # ------------------------------------------------------------------

    def build(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
    ) -> GraphResult:
        """
        Complete graph construction pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. Each row becomes one node.
        id_column : str, optional
            Column to use as node identifier.

        Returns
        -------
        GraphResult with NetworkX graph and metadata.
        """
        logger.info("=" * 60)
        logger.info("Graph Factory: Starting construction")
        logger.info(f"Input: {len(df)} rows, {len(df.columns)} columns")
        logger.info("=" * 60)

        # Step 1: Extract features
        id_col = id_column or self.config.id_column
        features = self.feature_extractor.extract(df, id_column=id_col)

        # Step 2: Build edges
        edge_strategy = self.config.edge_strategy
        edges: Dict[str, EdgeResult] = {}

        if edge_strategy == "knn":
            edge_result = self.edge_builder.knn_edges(features.X)
            edges["knn"] = edge_result

        elif edge_strategy == "epsilon":
            edge_result = self.edge_builder.epsilon_edges(features.X)
            edges["epsilon"] = edge_result

        elif edge_strategy == "shared_attr":
            edge_result = self.edge_builder.shared_attribute_edges(df)
            edges["shared_attr"] = edge_result

        elif edge_strategy == "heterogeneous":
            # Configure shared attributes from data
            cat_cols = features.cat_features
            if cat_cols and not self.config.edge_config.shared_attributes:
                self.config.edge_config.shared_attributes = cat_cols[:5]  # top 5 categoricals
            edges = self.edge_builder.heterogeneous_edges(features.X, df)

        else:
            raise ValueError(f"Unknown edge strategy: {edge_strategy}")

        # Step 3: Build NetworkX graph
        G, node_id_to_idx, idx_to_node_id = self._build_nx_graph(
            features, edges, df
        )

        total_edges = sum(e.n_edges for e in edges.values())

        logger.info("=" * 60)
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {total_edges} edges")
        logger.info(f"Edge types: {len(edges)}")
        for name, e in edges.items():
            logger.info(f"  - {name}: {e.n_edges} edges")
        logger.info("=" * 60)

        return GraphResult(
            graph=G,
            features=features,
            edges=edges,
            n_nodes=G.number_of_nodes(),
            n_edges=total_edges,
            n_edge_types=len(edges),
            node_id_to_idx=node_id_to_idx,
            idx_to_node_id=idx_to_node_id,
        )

    # ------------------------------------------------------------------
    # NetworkX Graph Assembly
    # ------------------------------------------------------------------

    def _build_nx_graph(
        self,
        features: FeatureResult,
        edges: Dict[str, EdgeResult],
        df: pd.DataFrame,
    ) -> Tuple[nx.Graph, Dict[str, int], Dict[int, str]]:
        """
        Assemble a NetworkX graph from features and edges.

        Each node stores:
        - 'features': its feature vector
        - 'feature_names': names of features
        - Original row data as node attributes
        """
        G = nx.Graph() if self.config.make_undirected else nx.DiGraph()

        n = len(features.node_ids)
        node_id_to_idx: Dict[str, int] = {}
        idx_to_node_id: Dict[int, str] = {}

        # Add nodes with features
        for i in range(n):
            node_id = str(features.node_ids[i])
            node_id_to_idx[node_id] = i
            idx_to_node_id[i] = node_id

            # Node attributes
            attrs = {
                "features": features.X[i],
                "feature_names": features.feature_names,
                "idx": i,
            }

            # Add original row data (first 100 chars for strings)
            if i < len(df):
                for col in df.columns:
                    val = df.iloc[i][col]
                    if isinstance(val, str) and len(val) > 100:
                        val = val[:100] + "..."
                    attrs[col] = val

            G.add_node(node_id, **attrs)

        # Add edges
        for edge_type_name, edge_result in edges.items():
            edge_arr = edge_result.edges
            weights = edge_result.edge_weights

            for e in range(edge_arr.shape[1]):
                src_idx = edge_arr[0, e]
                dst_idx = edge_arr[1, e]
                src_id = idx_to_node_id[src_idx]
                dst_id = idx_to_node_id[dst_idx]

                if self.config.remove_self_loops and src_id == dst_id:
                    continue

                G.add_edge(
                    src_id,
                    dst_id,
                    weight=float(weights[e]),
                    edge_type=edge_type_name,
                )

        return G, node_id_to_idx, idx_to_node_id

    # ------------------------------------------------------------------
    # Convenience: Build from CSV
    # ------------------------------------------------------------------

    def build_from_csv(
        self,
        csv_path: str,
        nrows: Optional[int] = None,
        **kwargs,
    ) -> GraphResult:
        """
        Load CSV and build graph in one call.

        Parameters
        ----------
        csv_path : str
            Path to CSV file.
        nrows : int, optional
            Number of rows to load (for large files).
        **kwargs : passed to pd.read_csv

        Returns
        -------
        GraphResult
        """
        logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path, nrows=nrows, **kwargs)
        return self.build(df)