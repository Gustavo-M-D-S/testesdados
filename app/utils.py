from pathlib import Path

from loguru import logger


def setup_logger() -> None:

    Path("outputs").mkdir(exist_ok=True)

    logger.add(
        "outputs/bankgraphai.log",
        rotation="20 MB",
        retention="10 days",
        level="INFO",
        enqueue=True,
    )