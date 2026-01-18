import functools

from . import _backend, graph
from .array import Array
from .graph import Graph, Node, Ops
from .symbolic import SymbolicArray

GRAPH_CACHE = {}


def _flatten(g, output):
    node_to_id = {node: i for i, node in enumerate(g.nodes)}
    flat_nodes = []
    for node in g.nodes:
        input_ids = [node_to_id[parent] for parent in node.inputs]
        flat_nodes.append((node.op, input_ids, node.shape, node.offset, node.strides))
    return flat_nodes, node_to_id[output.node]


def forge(fn):
    """Decorator"""

    @functools.wraps(fn)
    def wrapper(*args):
        input_metas = tuple((x.shape, x.offset, x.strides) for x in args)
        cache_key = (id(fn), input_metas)
        if cache_key in GRAPH_CACHE:
            backend_graph = GRAPH_CACHE[cache_key]
        else:
            print(f"Compiling func {fn.__name__}")
            g = Graph()
            graph.CURRENT_GRAPH = g
            sym_args = []
            for x in args:
                input_node = Node(Ops.INPUT, [], x.shape, x.offset, x.strides)
                g.add(input_node)
                sym_args.append(SymbolicArray(input_node))
            try:
                # assume for now that output is an Array type for fn
                # thus caught as a symbolicArray
                sym_out = fn(*sym_args)
            finally:
                graph.CURRENT_GRAPH = None

            flattened_graph = _flatten(g, sym_out)
            backend_graph = _backend.make_graph(flattened_graph)
            GRAPH_CACHE[cache_key] = backend_graph

        inputs = [x._handle for x in args]
        return Array(backend_graph.execute(inputs))

    return wrapper
