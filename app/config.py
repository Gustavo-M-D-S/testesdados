from pathlib import Path

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    name: str


class DataConfig(BaseModel):
    input_path: str
    output_path: str
    format: str


class SparkConfig(BaseModel):
    app_name: str
    master: str
    shuffle_partitions: int


class TrainingConfig(BaseModel):
    model: str
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_channels: int
    num_layers: int
    dropout: float
    mixed_precision: bool


class GraphConfig(BaseModel):
    neighbor_sizes: list[int]


class DashboardConfig(BaseModel):
    host: str
    port: int


class LoggingConfig(BaseModel):
    level: str


class Settings(BaseModel):
    project: ProjectConfig
    data: DataConfig
    spark: SparkConfig
    training: TrainingConfig
    graph: GraphConfig
    clustering: dict
    dashboard: DashboardConfig
    logging: LoggingConfig


def load_config(config_path: str = "configs/config.yaml") -> Settings:

    with Path(config_path).open(
        encoding="utf8"
    ) as f:

        data = yaml.safe_load(f)

    return Settings(**data)