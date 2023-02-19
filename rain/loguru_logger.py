"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

import sys

from loguru import logger

logger.remove()
logger.level("INFO", color="<white>")
logger.level("SUCCESS", color="<cyan>")

logger.add(
    sys.stdout,
    level="DEBUG",
    format="<yellow>{time}</yellow>|<level>{level}</level>|<blue>{extra[dataflow_id]}</blue>:<green>{extra[node_name]}</green>|{message}.",
    filter=lambda record: all(
        k in record["extra"] for k in ["dataflow_id", "node_name"]
    ),
)
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<yellow>{time}</yellow>|<level>{level}</level>|<blue>{extra[dataflow_id]}</blue>|{message}.",
    filter=lambda record: "node_name" not in record["extra"],
)
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<yellow>{time}</yellow>|<level>{level}</level>|<blue>{extra[node_name]}</blue>|{message}.",
    filter=lambda record: "dataflow_id" not in record["extra"]
    and "node_name" in record["extra"],
)
