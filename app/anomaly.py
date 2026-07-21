from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import torch

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from loguru import logger


# -------------------------
# RESULT
# -------------------------

@dataclass
class AnomalyResult:
    scores: np.ndarray
    labels: np.ndarray
    method: str


# -------------------------
# ANOMALY DETECTOR
# -------------------------

class AnomalyDetector:
    """
    Detects anomalies in transaction embeddings and graph representations.
    """

    def __init__(self):
        pass

    # -------------------------
    # ISOLATION FOREST
    # -------------------------

    def isolation_forest(self, X: np.ndarray) -> AnomalyResult:
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )

        model.fit(X)

        scores = model.decision_function(X)
        labels = model.predict(X)

        # -1 anomaly, 1 normal → convert
        labels = np.where(labels == -1, 1, 0)

        return AnomalyResult(scores, labels, "isolation_forest")

    # -------------------------
    # LOCAL OUTLIER FACTOR
    # -------------------------

    def lof(self, X: np.ndarray) -> AnomalyResult:
        model = LocalOutlierFactor(n_neighbors=20, contamination="auto")

        labels = model.fit_predict(X)

        scores = -model.negative_outlier_factor_

        labels = np.where(labels == -1, 1, 0)

        return AnomalyResult(scores, labels, "lof")

    # -------------------------
    # EMBEDDING DISTANCE ANOMALY
    # -------------------------

    def embedding_anomaly(self, X: np.ndarray) -> AnomalyResult:
        """
        Simple centroid-distance anomaly score.
        """

        centroid = X.mean(axis=0)

        distances = np.linalg.norm(X - centroid, axis=1)

        threshold = np.percentile(distances, 95)

        labels = (distances > threshold).astype(int)

        return AnomalyResult(distances, labels, "embedding_distance")

    # -------------------------
    # COMBINED SCORE
    # -------------------------

    def combined_score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Weighted combination of multiple anomaly detectors.
        """

        logger.info("Running anomaly ensemble...")

        iso = self.isolation_forest(X)
        lof = self.lof(X)
        emb = self.embedding_anomaly(X)

        combined_scores = (
            (iso.scores - iso.scores.min()) / (iso.scores.std() + 1e-8)
            + (lof.scores - lof.scores.min()) / (lof.scores.std() + 1e-8)
            + (emb.scores - emb.scores.min()) / (emb.scores.std() + 1e-8)
        )

        combined_scores = combined_scores / 3.0

        labels = (combined_scores > np.percentile(combined_scores, 95)).astype(int)

        logger.info("Anomaly detection completed.")

        return {
            "scores": combined_scores,
            "labels": labels,
        }