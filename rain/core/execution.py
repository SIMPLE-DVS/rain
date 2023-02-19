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

from rain.loguru_logger import logger
from rain.core.exception import CyclicDataFlowException


class Singleton(type):
    """Singleton class to represent all the possible executors available in Rain"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LocalExecutor(metaclass=Singleton):
    """A Local executor, meaning that the execution is performed on the machine that runs the Dataflow"""

    def execute(self, dataflow):
        """Method that executes the given Dataflow in a precise order. At each step it propagates the results to the
        following nodes by checking the edges.

        Parameters
        ----------
        dataflow : Dataflow
            The dataflow that has to be executed.
        """

        logger.debug("Checking if the Dataflow contains cycles", dataflow_id=dataflow.id)
        if not dataflow.is_acyclic():
            logger.critical("The Dataflow contains a cycle thus it can't be executed", dataflow_id=dataflow.id)
            raise CyclicDataFlowException(dataflow.id)

        ordered_nodes = dataflow.get_execution_ordered_nodes()

        for node in ordered_nodes:
            logger.info("Starting execution of the node", dataflow_id=dataflow.id, node_name=node)
            try:
                node.execute()
            except Exception as ex:
                logger.error(ex.__str__(), dataflow_id=dataflow.id, node_name=node)
                return False
            logger.success("Node executed succesfully", dataflow_id=dataflow.id, node_name=node)

            node_out_edge = dataflow.get_outgoing_edges(node)

            if not node_out_edge:
                continue

            for out_edge in node_out_edge:
                if (
                    len(out_edge.source.nodes_attributes) == 1
                    and len(out_edge.destination.nodes_attributes) == 1
                ):
                    out_edge.destination.node.set_input_value(
                        out_edge.destination.nodes_attributes[0],
                        out_edge.source.node.get_output_value(
                            out_edge.source.nodes_attributes[0]
                        ),
                    )
                elif (
                    len(out_edge.source.nodes_attributes) == 1
                    and len(out_edge.destination.nodes_attributes) > 1
                ):
                    for dest_inp in out_edge.destination.nodes_attributes:
                        out_edge.destination.node.set_input_value(
                            dest_inp,
                            out_edge.source.node.get_output_value(
                                out_edge.source.nodes_attributes[0]
                            ),
                        )
                elif (
                    len(out_edge.source.nodes_attributes) > 1
                    and len(out_edge.destination.nodes_attributes) > 1
                ):
                    for index, source_out in enumerate(out_edge.source_output):
                        out_edge.destination.node.set_input_value(
                            out_edge.destination.nodes_attributes[index],
                            out_edge.source.node.get_output_value(
                                out_edge.source.nodes_attributes[index]
                            ),
                        )
                else:
                    raise Exception("Error during values propagation!")

        logger.success("Execution completed", dataflow_id=dataflow.id)
        return True
