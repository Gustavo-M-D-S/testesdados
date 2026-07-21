from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch

from torch_geometric.explain import Explainer, GNNExplainer

from loguru import logger


# -------------------------
# RESULT
# -------------------------

@dataclass
class ExplanationResult:
    node_importance: np.ndarray
    edge_importance: np.ndarray
    summary: Dict[str, Any]


# -------------------------
# EXPLAINER ENGINE
# -------------------------

class GraphExplainerEngine:
    """
    Provides interpretability for GNN predictions and graph structure.
    """

    def __init__(self, model):
        self.model = model

        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=50),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
        )

    # -------------------------
    # NODE EXPLANATION
    # -------------------------

    def explain_node(self, x, edge_index, node_idx: int):
        """
        Explain prediction for a single node.
        """

        logger.info(f"Explaining node {node_idx}")

        explanation = self.explainer(
            x=x,
            edge_index=edge_index,
            index=node_idx,
        )

        node_mask = explanation.node_mask
        edge_mask = explanation.edge_mask

        return ExplanationResult(
            node_importance=node_mask.detach().cpu().numpy(),
            edge_importance=edge_mask.detach().cpu().numpy(),
            summary={
                "node_idx": node_idx,
                "top_features": self._top_features(node_mask),
            },
        )

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    def _top_features(self, mask: torch.Tensor, top_k: int = 10) -> List[int]:
        values, indices = torch.topk(mask, top_k)
        return indices.detach().cpu().tolist()

    # -------------------------
    # COMMUNITY SUMMARY
    # -------------------------

    def summarize_community(
        self,
        df_stats: Dict[str, Any],
        cluster_id: int,
    ) -> Dict[str, Any]:
        """
        Generates structured explanation of a community.
        """

        logger.info(f"Summarizing community {cluster_id}")

        summary = {
            "cluster_id": cluster_id,
            "num_transactions": df_stats.get("num_transactions", 0),
            "top_companies": df_stats.get("top_companies", [])[:5],
            "top_partners": df_stats.get("top_partners", [])[:5],
            "top_accounts": df_stats.get("top_accounts", [])[:5],
            "avg_transaction_value": df_stats.get("avg_value", 0.0),
            "dominant_currency": df_stats.get("currency", "unknown"),
            "time_span": {
                "start": df_stats.get("start_date"),
                "end": df_stats.get("end_date"),
            },
        }

        return summary

    # -------------------------
    # GRAPH LEVEL REPORT
    # -------------------------

    def generate_report(
        self,
        community_stats: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Creates global explanation report.
        """

        logger.info("Generating global explainability report...")

        report = {
            "num_communities": len(community_stats),
            "communities": community_stats,
            "risk_summary": {
                "high_risk_clusters": [
                    c for c in community_stats if c.get("risk_score", 0) > 0.8
                ],
            },
        }

        return report