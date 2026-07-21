from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    NumericType,
    TimestampType,
    BooleanType,
)

from loguru import logger


@dataclass
class PreprocessingResult:
    df: DataFrame
    feature_columns: Dict[str, List[str]]
    embedding_dim: int


class SparkPreprocessor:
    """
    Feature engineering pipeline for financial transaction graphs.

    All transformations are Spark-native (no pandas).
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = 384

    # -------------------------
    # TYPE INFERENCE
    # -------------------------

    def infer_column_types(self, df: DataFrame) -> Dict[str, str]:
        """
        Infer semantic column types using Spark schema + heuristics.
        """
        type_map = {}

        for field in df.schema.fields:
            name = field.name
            dtype = field.dataType

            if isinstance(dtype, StringType):
                if "date" in name or "time" in name:
                    type_map[name] = "timestamp"
                elif "id" in name:
                    type_map[name] = "id"
                elif "desc" in name or "name" in name:
                    type_map[name] = "text"
                else:
                    type_map[name] = "categorical"

            elif isinstance(dtype, NumericType):
                type_map[name] = "numeric"

            elif isinstance(dtype, TimestampType):
                type_map[name] = "timestamp"

            elif isinstance(dtype, BooleanType):
                type_map[name] = "boolean"

            else:
                type_map[name] = "unknown"

        return type_map

    # -------------------------
    # NULL HANDLING
    # -------------------------

    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Fill missing values based on column type.
        """
        for col_name in df.columns:
            dtype = df.schema[col_name].dataType

            if isinstance(dtype, NumericType):
                df = df.fillna({col_name: 0})
            else:
                df = df.fillna({col_name: "unknown"})

        return df

    # -------------------------
    # TEMPORAL FEATURES
    # -------------------------

    def add_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Extract temporal features from timestamp columns.
        """
        timestamp_cols = [
            c for c in df.columns if "date" in c or "time" in c
        ]

        for col_name in timestamp_cols:
            df = df.withColumn(
                f"{col_name}_year", F.year(F.col(col_name))
            ).withColumn(
                f"{col_name}_month", F.month(F.col(col_name))
            ).withColumn(
                f"{col_name}_day", F.dayofmonth(F.col(col_name))
            )

        return df

    # -------------------------
    # NUMERIC NORMALIZATION
    # -------------------------

    def normalize_numeric(self, df: DataFrame) -> DataFrame:
        """
        Z-score normalization for numeric columns.
        """
        numeric_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, NumericType)
        ]

        for col_name in numeric_cols:
            stats = df.select(
                F.mean(col_name).alias("mean"),
                F.stddev(col_name).alias("std"),
            ).collect()[0]

            mean = stats["mean"] or 0
            std = stats["std"] or 1

            df = df.withColumn(
                col_name,
                (F.col(col_name) - F.lit(mean)) / F.lit(std),
            )

        return df

    # -------------------------
    # TEXT EMBEDDINGS
    # -------------------------

    def add_text_embeddings(self, df: DataFrame) -> DataFrame:
        """
        Convert text columns into dense embeddings using SentenceTransformers.
        """

        text_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType)
            and ("desc" in f.name or "name" in f.name or "text" in f.name)
        ]

        if not text_cols:
            return df

        def embed_text(texts: List[str]) -> List[List[float]]:
            model = self.text_model
            return model.encode(texts).tolist()

        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, FloatType

        embed_udf = udf(
            lambda x: self.text_model.encode([x])[0].tolist(),
            ArrayType(FloatType()),
        )

        for col_name in text_cols:
            df = df.withColumn(
                f"{col_name}_embedding",
                embed_udf(F.col(col_name)),
            )

        return df

    # -------------------------
    # HIGH CARDINALITY ENCODING
    # -------------------------

    def hash_encode(self, df: DataFrame) -> DataFrame:
        """
        Hash-based encoding for categorical features (no OneHot).
        """
        categorical_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType)
        ]

        for col_name in categorical_cols:
            df = df.withColumn(
                f"{col_name}_hash",
                F.hash(F.col(col_name)) % 10000,
            )

        return df

    # -------------------------
    # ENTITY AGGREGATIONS
    # -------------------------

    def add_entity_aggregations(self, df: DataFrame) -> DataFrame:
        """
        Aggregate transactional patterns per entity.
        """

        if "company" in df.columns:
            agg = df.groupBy("company").agg(
                F.count("*").alias("company_tx_count"),
                F.avg(F.col("movement")).alias("company_avg_movement"),
            )

            df = df.join(agg, on="company", how="left")

        if "partner" in df.columns:
            agg = df.groupBy("partner").agg(
                F.count("*").alias("partner_tx_count"),
            )

            df = df.join(agg, on="partner", how="left")

        return df

    # -------------------------
    # FULL PIPELINE
    # -------------------------

    def transform(self, df: DataFrame) -> PreprocessingResult:
        """
        Full preprocessing pipeline.
        """

        logger.info("Starting preprocessing pipeline...")

        type_map = self.infer_column_types(df)

        df = self.handle_missing_values(df)
        df = self.add_temporal_features(df)
        df = self.normalize_numeric(df)
        df = self.hash_encode(df)
        df = self.add_text_embeddings(df)
        df = self.add_entity_aggregations(df)

        feature_columns = {
            "numeric": [
                f.name for f in df.schema.fields
                if isinstance(f.dataType, NumericType)
            ],
            "categorical": [
                f.name for f in df.schema.fields
                if isinstance(f.dataType, StringType)
            ],
        }

        logger.info("Preprocessing completed successfully.")

        return PreprocessingResult(
            df=df,
            feature_columns=feature_columns,
            embedding_dim=self.embedding_dim,
        )