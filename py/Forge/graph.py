class Ops:
    INPUT = 0
    MATMUL = 1
    ADD = 2
    MUL = 3
    DIV = 4
    SUB = 5


class Node:
    __slots__ = ("op", "inputs", "shape")

    def __init__(self, op: int, inputs: list, shape: tuple):
        self.op = op
        self.inputs = inputs
        self.shape = shape


class Graph:
    def __init__(self):
        self.nodes = []

    def add(self, node: Node):
        self.nodes.append(node)
        return node


CURRENT_GRAPH = None
