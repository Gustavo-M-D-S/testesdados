"""
Community Explainer Module
==========================
Explains why each node belongs to its community.

Provides:
1. Feature importance per community (Δ from global mean)
2. Top distinguishing features per community
3. Community profile (statistical summary)
4. Node-level explanation (which features influenced assignment)
5. Global report generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx

from loguru import logger


@dataclass
class CommunityProfile:
    """Statistical profile of a single community."""

    community_id: int
    size: int
    feature_importance: Dict[str, float]  # feature_name -> Δ from global mean
    top_features: List[Tuple[str, float]]  # sorted by importance
    numerical_stats: Dict[str, Dict[str, float]]  # feature -> {mean, std, min, max}
    categorical_modes: Dict[str, str]  # feature -> most common value
    sample_nodes: List[str]  # representative node IDs


@dataclass
class NodeExplanation:
    """Explanation for a single node's community assignment."""

    node_id: str
    community_id: int
    feature_influences: Dict[str, float]  # feature_name -> influence score
    top_influences: List[Tuple[str, float]]  # sorted by influence
    distance_to_centroid: float  # Euclidean distance to community centroid
    is_representative: bool  # True if close to centroid


@dataclass
class ExplanationReport:
    """Complete explainability report."""

    n_communities: int
    n_nodes: int
    communities: Dict[int, CommunityProfile]
    global_feature_importance: Dict[str, float]
    method: str
    quality_score: float


class CommunityExplainer:
    """
    Explains community detection results.

    Usage:
        explainer = CommunityExplainer()
        report = explainer.explain(G, community_labels, feature_names)
        node_exp = explainer.explain_node(G, "node_123", community_labels, feature_names)
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Community Profiles
    # ------------------------------------------------------------------

    def profile_community(
        self,
        G: nx.Graph,
        community_id: int,
        node_indices: List[int],
        all_features: np.ndarray,
        feature_names: List[str],
        original_df: Optional[pd.DataFrame] = None,
    ) -> CommunityProfile:
        """
        Generate a statistical profile of a single community.

        Parameters
        ----------
        G : nx.Graph
            The graph (used for structural metrics).
        community_id : int
            Community label.
        node_indices : List[int]
            Indices of nodes in this community.
        all_features : np.ndarray
            Full feature matrix (N x d).
        feature_names : List[str]
            Names of each feature.
        original_df : pd.DataFrame, optional
            Original DataFrame for categorical mode computation.

        Returns
        -------
        CommunityProfile
        """
        if len(node_indices) == 0:
            return CommunityProfile(
                community_id=community_id,
                size=0,
                feature_importance={},
                top_features=[],
                numerical_stats={},
                categorical_modes={},
                sample_nodes=[],
            )

        # Community features
        comm_features = all_features[node_indices]
        global_mean = all_features.mean(axis=0)
        comm_mean = comm_features.mean(axis=0)

        # Feature importance = absolute Δ from global mean
        importance = np.abs(comm_mean - global_mean)
        feature_importance = {
            feature_names[i]: float(importance[i])
            for i in range(len(feature_names))
        }

        # Top features sorted by importance
        sorted_idx = np.argsort(importance)[::-1]
        top_features = [
            (feature_names[i], float(importance[i]))
            for i in sorted_idx[:10]  # top 10
            if importance[i] > 0
        ]

        # Numerical stats
        numerical_stats = {}
        for i, fname in enumerate(feature_names):
            numerical_stats[fname] = {
                "mean": float(comm_features[:, i].mean()),
                "std": float(comm_features[:, i].std()),
                "min": float(comm_features[:, i].min()),
                "max": float(comm_features[:, i].max()),
            }

        # Categorical modes (from original DataFrame)
        categorical_modes = {}
        if original_df is not None:
            node_ids = [list(G.nodes())[idx] for idx in node_indices]
            comm_df = original_df.loc[
                original_df.index.isin(node_indices)
                if isinstance(original_df.index, pd.RangeIndex)
                else original_df.iloc[node_indices]
            ]
            for col in original_df.columns:
                if original_df[col].dtype == object:
                    categorical_modes[col] = str(comm_df[col].mode().iloc[0]) if not comm_df[col].mode().empty else ""

        # Sample nodes (up to 5 closest to centroid)
        distances = np.linalg.norm(comm_features - comm_mean, axis=1)
        closest_idx = np.argsort(distances)[:5]
        nodes_list = list(G.nodes())
        sample_nodes = [str(nodes_list[node_indices[i]]) for i in closest_idx]

        return CommunityProfile(
            community_id=community_id,
            size=len(node_indices),
            feature_importance=feature_importance,
            top_features=top_features,
            numerical_stats=numerical_stats,
            categorical_modes=categorical_modes,
            sample_nodes=sample_nodes,
        )

    # ------------------------------------------------------------------
    # Full Explanation Report
    # ------------------------------------------------------------------

    def explain(
        self,
        G: nx.Graph,
        community_labels: np.ndarray,
        feature_names: List[str],
        original_df: Optional[pd.DataFrame] = None,
        method: str = "leiden",
        quality_score: float = 0.0,
    ) -> ExplanationReport:
        """
        Generate a complete explainability report.

        Parameters
        ----------
        G : nx.Graph
            The graph.
        community_labels : np.ndarray
            (N,) array of community labels for each node.
        feature_names : List[str]
            Names of features.
        original_df : pd.DataFrame, optional
            Original data for categorical context.
        method : str
            Community detection method name.
        quality_score : float
            Quality score (modularity, silhouette).

        Returns
        -------
        ExplanationReport
        """
        logger.info("Generating explainability report...")

        # Extract feature matrix from graph nodes
        nodes = list(G.nodes())
        all_features = np.array([G.nodes[n].get("features", np.zeros(len(feature_names))) for n in nodes])

        # Group nodes by community
        unique_labels = sorted(set(community_labels))
        communities: Dict[int, CommunityProfile] = {}

        for label in unique_labels:
            node_indices = [i for i, l in enumerate(community_labels) if l == label]
            profile = self.profile_community(
                G, int(label), node_indices, all_features, feature_names, original_df
            )
            communities[int(label)] = profile

        # Global feature importance
        global_importance = {}
        for fname in feature_names[:20]:  # top 20 features
            # Variance of community means for this feature
            comm_means = [
                communities[cid].numerical_stats.get(fname, {}).get("mean", 0)
                for cid in communities
            ]
            global_importance[fname] = float(np.std(comm_means)) if len(comm_means) > 1 else 0.0

        # Sort global importance
        global_importance = dict(
            sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        report = ExplanationReport(
            n_communities=len(communities),
            n_nodes=len(nodes),
            communities=communities,
            global_feature_importance=global_importance,
            method=method,
            quality_score=quality_score,
        )

        logger.info(f"Report: {report.n_communities} communities, {report.n_nodes} nodes")

        return report

    # ------------------------------------------------------------------
    # Node-Level Explanation
    # ------------------------------------------------------------------

    def explain_node(
        self,
        G: nx.Graph,
        node_id: str,
        community_labels: np.ndarray,
        feature_names: List[str],
    ) -> NodeExplanation:
        """
        Explain why a specific node belongs to its community.

        Computes which features most strongly influenced its assignment
        by measuring how far the node is from its community centroid
        in each feature dimension.
        """
        nodes = list(G.nodes())

        if node_id not in nodes:
            raise ValueError(f"Node {node_id} not found in graph")

        node_idx = nodes.index(node_id)
        community_id = int(community_labels[node_idx])

        # Node features
        node_feats = G.nodes[node_id].get("features")
        if node_feats is None:
            raise ValueError(f"Node {node_id} has no features")

        # Community features
        comm_indices = [i for i, l in enumerate(community_labels) if l == community_id]
        all_features = np.array([G.nodes[nodes[i]].get("features", np.zeros(len(feature_names))) for i in comm_indices])
        comm_centroid = all_features.mean(axis=0)
        comm_std = all_features.std(axis=0) + 1e-8

        # Feature influence = how far is this node from centroid, in std units
        influence = np.abs(node_feats - comm_centroid) / comm_std

        feature_influences = {
            feature_names[i]: float(influence[i])
            for i in range(len(feature_names))
        }

        sorted_idx = np.argsort(influence)[::-1]
        top_influences = [
            (feature_names[i], float(influence[i]))
            for i in sorted_idx[:10]
        ]

        distance_to_centroid = float(np.linalg.norm(node_feats - comm_centroid))

        # Is representative? (within 1 std of centroid)
        is_representative = distance_to_centroid < comm_std.mean()

        return NodeExplanation(
            node_id=node_id,
            community_id=community_id,
            feature_influences=feature_influences,
            top_influences=top_influences,
            distance_to_centroid=distance_to_centroid,
            is_representative=is_representative,
        )

    # ------------------------------------------------------------------
    # Report to DataFrame
    # ------------------------------------------------------------------

    def report_to_dataframe(self, report: ExplanationReport) -> pd.DataFrame:
        """
        Convert explanation report to a pandas DataFrame for display.
        """
        rows = []
        for cid, profile in report.communities.items():
            top_feat_str = "; ".join([f"{name}: {val:.3f}" for name, val in profile.top_features[:5]])
            rows.append({
                "Community": cid,
                "Size": profile.size,
                "Top Features": top_feat_str,
                "Sample Nodes": ", ".join(profile.sample_nodes[:3]),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Generate Summaries
    # ------------------------------------------------------------------

    def generate_summary(self, report: ExplanationReport) -> str:
        """
        Generate a human-readable summary of the explainability report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("COMMUNITY DETECTION EXPLANATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Method: {report.method}")
        lines.append(f"Quality Score: {report.quality_score:.4f}")
        lines.append(f"Total Communities: {report.n_communities}")
        lines.append(f"Total Nodes: {report.n_nodes}")
        lines.append("")

        lines.append("-- Global Feature Importance (top features that distinguish communities) --")
        for fname, imp in report.global_feature_importance.items():
            lines.append(f"  {fname}: {imp:.4f}")
        lines.append("")

        for cid, profile in report.communities.items():
            lines.append(f"\nCommunity {cid} (size: {profile.size}):")
            lines.append(f"  Top distinguishing features:")
            for name, val in profile.top_features[:5]:
                lines.append(f"    - {name}: Δ = {val:.4f}")
            lines.append(f"  Sample nodes: {', '.join(profile.sample_nodes[:3])}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)