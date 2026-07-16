import pytest

from pyspark.sql import SparkSession

from app.preprocessing import SparkPreprocessor
from app.graph_builder import HeteroGraphBuilder


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("test-graph")
        .getOrCreate()
    )


def test_graph_building(spark):

    data = [
        ("1", "COMP_A", "PARTNER_X", 100.0),
        ("2", "COMP_B", "PARTNER_Y", 200.0),
        ("3", "COMP_A", "PARTNER_X", 150.0),
    ]

    df = spark.createDataFrame(
        data,
        ["id", "company", "partner", "movement"],
    )

    preprocessor = SparkPreprocessor()
    processed = preprocessor.transform(df)

    builder = HeteroGraphBuilder()

    result = builder.build(processed.df)

    graph = result.data

    assert "transaction" in graph.node_types
    assert len(graph.edge_types) > 0
    assert graph["transaction"].num_nodes > 0