from graphviz import Digraph
from typing import Set, Tuple

from engine.value import Value


class Visualize:
    """
    A utility class for visualizing a computational graph composed of Node objects

    Uses the `graphviz` library to generate a visual representation of the graph,
    showing how nodes are connected through operations
    """

    @staticmethod
    def _trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
        """
        Traces the computational graph starting from the root node

        Args
            root `Node`: The root node from which to begin the trace

        Returns
            `Tuple[Set[Node], Set[Tuple[Node, Node]]]`:
                A tuple containing:
                    - a set of all nodes in the graph
                    - a set of edges representing dependencies between nodes
        """

        nodes: Set[Value] = set()
        edges: Set[Tuple[Value, Value]] = set()

        def build(v: Value) -> None:
            if v not in nodes:
                nodes.add(v)
                for child in v._previous:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    @staticmethod
    def _format_node_label(node: Value) -> str:
        """
        Generates a formatted label for a node for visualization purposes

        Args
            node `Node`: The node to format

        Returns:
            `str`: A Graphviz label string including optional label, data, and gradient
        """

        parts = []

        if node._label is not None:
            parts.append(node._label)

        parts.append(f"data {node.data:.4f}")
        parts.append(f"grad {node.grad:.4f}")

        return f"{{ {' | '.join(parts)} }}"

    @staticmethod
    def draw(root: Value, format: str = "svg", rankdir: str = "LR") -> Digraph:
        """
        Draws the computational graph rooted at the specified node

        Args:
            root `Node`: The root node of the computational graph
            format `str`: Output format for the graph (default is "svg")
            rankdir `str`: Layout direction; either "LR" (left-to-right) or "TB" (top-to-bottom)

        Returns:
            `Digraph`: A Graphviz Digraph object representing the graph.
        """

        assert rankdir in {"LR", "TB"}, "rankdir must be 'LR' or 'TB'"

        nodes, edges = Visualize._trace(root)
        dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

        for n in nodes:
            node_id = str(id(n))
            label = Visualize._format_node_label(n)
            dot.node(name=node_id, label=label, shape="record")

            if n._operation is not None:
                if isinstance(n._operation, tuple):
                    op, suffix = n._operation
                    op_label = f"{op.value}{suffix}" if suffix else op.value
                else:
                    op_label = n._operation.value

                op_node_name = f"op_{id(n)}"
                dot.node(name=op_node_name, label=op_label)
                dot.edge(op_node_name, node_id)

        for parent, child in edges:
            if child._operation is not None:
                op_node_name = f"op_{id(child)}"
                dot.edge(str(id(parent)), op_node_name)

        return dot
