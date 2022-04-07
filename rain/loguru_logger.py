import sys

from loguru import logger

logger.remove()
logger.level("INFO", color="<white>")
logger.level("SUCCESS", color="<cyan>")

logger.add(
    sys.stdout,
    level="DEBUG",
    format="<yellow>{time}</yellow>:<level>{level}</level>:<blue>{extra[dataflow_id]}</blue>:<green>{extra[node_name]}</green>: {message}.",
    filter=lambda record: all(
        k in record["extra"] for k in ["dataflow_id", "node_name"]
    ),
)
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<yellow>{time}</yellow>:<level>{level}</level>:<blue>{extra[dataflow_id]}</blue>: {message}.",
    filter=lambda record: "node_name" not in record["extra"],
)
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<yellow>{time}</yellow>:<level>{level}</level>:<blue>{extra[node_name]}</blue>: {message}.",
    filter=lambda record: "dataflow_id" not in record["extra"]
    and "node_name" in record["extra"],
)
