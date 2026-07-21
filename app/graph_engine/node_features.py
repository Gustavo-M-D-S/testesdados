"""
Node Feature Extraction Module
================================
Converts each row of a DataFrame into a feature vector for graph nodes.

Supports:
- Numerical features (z-score normalization)
- Categorical features (one-hot or hash encoding)
- Text features (SentenceTransformer embeddings)
- Temporal features (cyclic encoding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from loguru import logger


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    one_hot_max_categories: int = 20
    hash_modulus: int = 10000
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_dim: int = 384
    use_text_embeddings: bool = False  # disabled by default for speed


@dataclass
class FeatureResult:
    """Result of feature extraction."""

    X: np.ndarray  # Feature matrix (N x d)
    feature_names: List[str]  # Names of each feature column
    node_ids: np.ndarray  # Original row identifiers
    num_features: int
    cat_features: List[str]
    text_features: List[str]
    time_features: List[str]


class FeatureExtractor:
    """
    Converts a pandas DataFrame into a numerical feature matrix.

    Each row becomes one node's feature vector. The extractor:
    1. Identifies column types (numeric, categorical, text, temporal)
    2. Encodes each type appropriately
    3. Concatenates into a single feature matrix
    4. Normalizes numerical columns
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._text_model = None

    # ------------------------------------------------------------------
    # Column Type Inference
    # ------------------------------------------------------------------

    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer semantic column types from a pandas DataFrame.

        Returns a dict mapping column name -> type:
        'numeric', 'categorical', 'text', 'temporal', 'id', 'boolean', 'unknown'
        """
        type_map: Dict[str, str] = {}

        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype

            # Detect ID columns
            if "id" in col_lower and dtype == object:
                type_map[col] = "id"
                continue

            # Detect temporal columns
            if any(x in col_lower for x in ["date", "time", "timestamp", "ano", "mes", "dia"]):
                type_map[col] = "temporal"
                continue

            # Detect text columns
            if any(x in col_lower for x in ["desc", "name", "text", "obs", "historico"]):
                type_map[col] = "text"
                continue

            # Numeric types
            if np.issubdtype(dtype, np.number):
                type_map[col] = "numeric"
                continue

            # Boolean
            if dtype == bool:
                type_map[col] = "boolean"
                continue

            # String / object → categorical
            if dtype == object:
                type_map[col] = "categorical"
                continue

            type_map[col] = "unknown"

        return type_map

    # ------------------------------------------------------------------
    # Numerical Encoding
    # ------------------------------------------------------------------

    def encode_numeric(self, df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Z-score normalize numerical columns."""
        if not cols:
            return np.empty((len(df), 0)), []

        X_num = df[cols].values.astype(np.float64)
        means = np.nanmean(X_num, axis=0)
        stds = np.nanstd(X_num, axis=0) + 1e-8
        X_norm = (X_num - means) / stds
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        return X_norm, cols

    # ------------------------------------------------------------------
    # Categorical Encoding
    # ------------------------------------------------------------------

    def encode_categorical(
        self, df: pd.DataFrame, cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode categorical columns.
        - Low cardinality (<= threshold): one-hot encoding
        - High cardinality: hash encoding
        """
        if not cols:
            return np.empty((len(df), 0)), []

        encoded_parts: List[np.ndarray] = []
        feature_names: List[str] = []

        for col in cols:
            unique_vals = df[col].nunique()

            if unique_vals <= self.config.one_hot_max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                encoded_parts.append(dummies.values.astype(np.float64))
                feature_names.extend(dummies.columns.tolist())
            else:
                # Hash encoding
                hashed = df[col].apply(lambda x: hash(str(x)) % self.config.hash_modulus)
                # Normalize to [0, 1]
                normalized = hashed.values.astype(np.float64) / self.config.hash_modulus
                encoded_parts.append(normalized.reshape(-1, 1))
                feature_names.append(f"{col}_hash")

        if encoded_parts:
            return np.column_stack(encoded_parts), feature_names
        return np.empty((len(df), 0)), []

    # ------------------------------------------------------------------
    # Temporal Encoding
    # ------------------------------------------------------------------

    def encode_temporal(self, df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Encode temporal columns using cyclic features.

        For each date column, extracts:
        - sin(2π * day / 365), cos(2π * day / 365)
        - sin(2π * month / 12), cos(2π * month / 12)
        - day of week (0-6) normalized
        """
        if not cols:
            return np.empty((len(df), 0)), []

        encoded_parts: List[np.ndarray] = []
        feature_names: List[str] = []

        for col in cols:
            try:
                dates = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                continue

            valid = dates.notna()
            day_of_year = np.where(valid, dates.dt.dayofyear.values, 0)
            month = np.where(valid, dates.dt.month.values, 0)
            day_of_week = np.where(valid, dates.dt.dayofweek.values, 0)

            features = np.column_stack([
                np.sin(2 * np.pi * day_of_year / 365),
                np.cos(2 * np.pi * day_of_year / 365),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                day_of_week / 6.0,  # normalize to [0, 1]
            ])

            encoded_parts.append(features)
            feature_names.extend([
                f"{col}_sin_day",
                f"{col}_cos_day",
                f"{col}_sin_month",
                f"{col}_cos_month",
                f"{col}_dow_norm",
            ])

        if encoded_parts:
            return np.column_stack(encoded_parts), feature_names
        return np.empty((len(df), 0)), []

    # ------------------------------------------------------------------
    # Text Encoding
    # ------------------------------------------------------------------

    def _load_text_model(self):
        """Lazy-load SentenceTransformer."""
        if self._text_model is None and self.config.use_text_embeddings:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self.config.text_model_name)
        return self._text_model

    def encode_text(self, df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Encode text columns using SentenceTransformer embeddings.
        Returns a (N x embedding_dim) matrix.
        """
        if not cols or not self.config.use_text_embeddings:
            return np.empty((len(df), 0)), []

        model = self._load_text_model()
        if model is None:
            return np.empty((len(df), 0)), []

        # Concatenate all text columns into a single string per row
        texts = df[cols].fillna("").agg(" ".join, axis=1).tolist()

        logger.info(f"Encoding {len(texts)} text rows with SentenceTransformer...")
        embeddings = model.encode(texts, show_progress_bar=False)
        logger.info(f"Text encoding done: shape {embeddings.shape}")

        feature_names = [f"text_emb_{i}" for i in range(embeddings.shape[1])]
        return embeddings, feature_names

    # ------------------------------------------------------------------
    # Main Extraction Pipeline
    # ------------------------------------------------------------------

    def extract(self, df: pd.DataFrame, id_column: Optional[str] = None) -> FeatureResult:
        """
        Full feature extraction pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. Each row becomes one node.
        id_column : str, optional
            Column to use as node identifier. If None, uses the index.

        Returns
        -------
        FeatureResult with feature matrix X, feature names, and node IDs.
        """
        logger.info(f"Extracting features from {len(df)} rows x {len(df.columns)} columns")

        # Infer types
        type_map = self.infer_column_types(df)
        logger.debug(f"Column types: {type_map}")

        # Separate columns by type
        id_cols = [c for c, t in type_map.items() if t == "id"]
        num_cols = [c for c, t in type_map.items() if t == "numeric"]
        cat_cols = [c for c, t in type_map.items() if t == "categorical"]
        text_cols = [c for c, t in type_map.items() if t == "text"]
        time_cols = [c for c, t in type_map.items() if t == "temporal"]
        bool_cols = [c for c, t in type_map.items() if t == "boolean"]

        # Node IDs
        if id_column and id_column in df.columns:
            node_ids = df[id_column].values
        elif id_cols:
            node_ids = df[id_cols[0]].values
        else:
            node_ids = np.arange(len(df))

        # Encode each type
        X_num, num_names = self.encode_numeric(df, num_cols)
        X_cat, cat_names = self.encode_categorical(df, cat_cols)
        X_time, time_names = self.encode_temporal(df, time_cols)
        X_text, text_names = self.encode_text(df, text_cols)

        # Boolean → float
        if bool_cols:
            X_bool = df[bool_cols].values.astype(np.float64)
            bool_names = bool_cols
        else:
            X_bool = np.empty((len(df), 0))
            bool_names = []

        # Concatenate all feature matrices
        parts = [X_num, X_cat, X_time, X_text, X_bool]
        non_empty = [p for p in parts if p.shape[1] > 0]

        if non_empty:
            X = np.column_stack(non_empty)
        else:
            X = np.empty((len(df), 0))

        all_names = num_names + cat_names + time_names + text_names + bool_names

        logger.info(
            f"Feature matrix: {X.shape} "
            f"(num={len(num_names)}, cat={len(cat_names)}, "
            f"time={len(time_names)}, text={len(text_names)}, bool={len(bool_names)})"
        )

        return FeatureResult(
            X=X,
            feature_names=all_names,
            node_ids=node_ids,
            num_features=len(num_names),
            cat_features=cat_names,
            text_features=text_names,
            time_features=time_names,
        )