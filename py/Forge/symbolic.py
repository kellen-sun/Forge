from . import graph
from .graph import Node, Ops


class SymbolicArray:
    def __init__(self, node: Node):
        self.node = node
        self.shape = node.shape

    def __add__(self, other):
        new_node = Node(Ops.ADD, [self.node, other.node], self.shape)
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)

    def __matmul__(self, other):
        new_node = Node(
            "matmul", [self.node, other.node], (self.shape[-2], other.shape[-1])
        )
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)
