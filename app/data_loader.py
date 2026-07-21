from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

from loguru import logger


SUPPORTED_FORMATS = {"csv", "parquet", "orc", "delta", "auto"}


@dataclass
class DataLoadResult:
    df: DataFrame
    format: str
    schema: StructType
    path: str


class SparkDataLoader:
    """
    Responsible for ingesting large-scale datasets using Apache Spark
    with automatic format detection and safe reading.
    """

    def __init__(self, app_name: str, master: str, shuffle_partitions: int):
        self.app_name = app_name
        self.master = master
        self.shuffle_partitions = shuffle_partitions
        self.spark = self._create_spark_session()

    def _create_spark_session(self) -> SparkSession:
        logger.info("Creating SparkSession...")

        import sys

        spark = (
            SparkSession.builder.appName(self.app_name)
            .master(self.master)
            .config("spark.sql.shuffle.partitions", str(self.shuffle_partitions))
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        return spark

    # -----------------------------
    # FORMAT DETECTION
    # -----------------------------
    def detect_format(self, path: str) -> str:
        """
        Detect file format from file or directory.
        """
        p = Path(path)

        # Single file
        if p.is_file():
            return self._format_from_suffix(p.suffix)

        # Directory → inspect contents
        if p.is_dir():
            files = list(p.glob("*"))

            if any(f.suffix == ".csv" for f in files):
                return "csv"
            if any(f.suffix == ".parquet" for f in files):
                return "parquet"
            if any(f.suffix == ".orc" for f in files):
                return "orc"
            if any("delta" in str(f) for f in files):
                return "delta"

        return "parquet"

    def _format_from_suffix(self, suffix: str) -> str:
        suffix = suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix == ".parquet":
            return "parquet"
        if suffix == ".orc":
            return "orc"
        return "parquet"

    # -----------------------------
    # PUBLIC READ METHOD
    # -----------------------------
    def read(self, path: str, fmt: str = "auto") -> DataLoadResult:
        logger.info(f"Loading dataset from: {path}")

        if fmt == "auto":
            fmt = self.detect_format(path)

        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}")

        # IMPORTANT: ensure path exists
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if fmt == "csv":
            df = self._read_csv(path)
        elif fmt == "parquet":
            df = self._read_parquet(path)
        elif fmt == "orc":
            df = self._read_orc(path)
        elif fmt == "delta":
            df = self._read_delta(path)
        else:
            raise ValueError(f"Invalid format: {fmt}")

        df = self._standardize_columns(df)

        logger.info(f"Loaded dataset with {len(df.columns)} columns")

        return DataLoadResult(
            df=df,
            format=fmt,
            schema=df.schema,
            path=path,
        )

    # -----------------------------
    # READERS
    # -----------------------------
    def _read_csv(self, path: str) -> DataFrame:
        # Works for both file and folder
        res = (
            self.spark.read.option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("sep", ",")
            .csv(path)
        )

        print('reading finished')
        return res

    def _read_parquet(self, path: str) -> DataFrame:
        return self.spark.read.parquet(path)

    def _read_orc(self, path: str) -> DataFrame:
        return self.spark.read.orc(path)

    def _read_delta(self, path: str) -> DataFrame:
        return self.spark.read.format("delta").load(path)

    # -----------------------------
    # CLEANING
    # -----------------------------
    def _standardize_columns(self, df: DataFrame) -> DataFrame:
        """
        Normalize column names:
        - lowercase
        - spaces → underscores
        - remove hyphens
        """
        for col_name in df.columns:
            new_name = (
                col_name.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            df = df.withColumnRenamed(col_name, new_name)

        return df

    # -----------------------------
    # DEBUG
    # -----------------------------
    def show_sample(self, df: DataFrame, n: int = 5) -> None:
        logger.info("Showing dataset sample...")
        df.show(n, truncate=False)