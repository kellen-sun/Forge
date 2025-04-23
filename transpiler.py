import inspect
import ast
import logging
import custom_metal

def setup_logging():
    logger = logging.getLogger("metal_transpiler")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # File handler
    fh = logging.FileHandler("transpiler.log", mode='w')
    fh.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

class TypeAnnotator(ast.NodeVisitor):
    def __init__(self, var_types):
        self.var_types = var_types
        self.builtins = {
            'sum': 'scalar',
            'len': 'scalar'
        }

    def visit_Name(self, node):
        node.inferred_type = self.var_types.get(node.id, 'unknown')

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        ltype = getattr(node.left, 'inferred_type', 'unknown')
        rtype = getattr(node.right, 'inferred_type', 'unknown')
        if ltype == rtype:
            node.inferred_type = ltype
        elif 'array' in (ltype[0], rtype[0]):
            node.inferred_type = ltype if ltype[0] == 'array' else rtype
        else:
            node.inferred_type = 'scalar'
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.builtins:
                node.inferred_type = self.builtins[func_name]
            else:
                node.inferred_type = 'unknown'
        self.generic_visit(node)

def get_func_args(tree):
    args = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if isinstance(arg.arg, str):
                    args.append(arg.arg)
    return args

def get_args(tree):
    args = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if isinstance(arg.arg, str):
                    args.append(arg.arg)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    args.append(target.id)
    logger.info(f"args: {args}")
    return args

def get_func_name(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            logger.info(f"func_name: {node.name}")
            return node.name
    return None

def is_scalar_constant(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_expr(node):
    global runtime_items
    global buffer_lookup
    if isinstance(node, ast.BinOp):
        left = get_expr(node.left)
        right = get_expr(node.right)
        if is_scalar_constant(left) and is_scalar_constant(right):
            l, r = float(left), float(right)
            if isinstance(node.op, ast.Add):
                return str(l + r)
            elif isinstance(node.op, ast.Sub):
                return str(l - r)
            elif isinstance(node.op, ast.Mult):
                return str(l * r)
            elif isinstance(node.op, ast.Div):
                return str(l / r)
        op = node.op
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }
        if op_map.get(type(op)):
            return f"{left} {op_map[type(op)]} {right}"
        else:
            logger.error(f"Unsupported binary operator: {ast.dump(op)}")
            raise NotImplementedError(f"Unsupported binary operator: {ast.dump(op)}")
    elif isinstance(node, ast.Name):
        return f"{node.id}[id]"
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name == "sum":
            return custom_metal.metal_sum(node, buffer_lookup, runtime_items)
        elif func_name == "len":
            return str(node.args[0].inferred_type[1])
        else:
            logger.error(f"Unsupported function call: {ast.dump(node)}")
            raise NotImplementedError(f"Unsupported function call: {ast.dump(node)}")

    else:
        logger.error(f"Unsupported expression: {ast.dump(node)}")
        raise NotImplementedError(f"Unsupported expression: {ast.dump(node)}")

def get_body(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body
            for stmt in body:
                if isinstance(stmt, ast.Return):
                    returnVal = get_expr(stmt.value)
                    if is_scalar_constant(returnVal):
                        return returnVal
                    out.append(f"out[id] = {returnVal};")
                elif isinstance(stmt, ast.Expr):
                    # should check if this has side effect in the future
                    out.append(f"{get_expr(stmt)};")           
                elif isinstance(stmt, ast.Assign):
                    # what if there are multiple targets?
                    out.append(f"{stmt.targets[0].id}[id] = {get_expr(stmt.value)};")
                else:
                    logger.error(f"Unsupported statement: {ast.dump(stmt)}")
                    raise NotImplementedError(f"Unsupported statement: {ast.dump(stmt)}")
    logger.info(f"body: {out}")
    return "\n    ".join(out)

def transpile(tree, source: str, rt_items) -> str:
    global runtime_items
    global buffer_lookup
    runtime_items = rt_items
    n = runtime_items[2]
    
    TypeAnnotator({'a': ('array', n), 'b': ('array', n)}).visit(tree)
    logger.info(f"AST:\n{ast.dump(tree, indent=4)}")

    args = get_args(tree)
    buf_decls = "\n    ".join([
        f"device float* {arg} [[ buffer({3+i}) ]]," for i, arg in enumerate(args)
    ])
    buffer_lookup = {"out": 0, "tempA": 1, "tempB": 2}
    for i, arg in enumerate(args):
        buffer_lookup[arg] = i + 3
    logger.info(f"buffer_lookup: {buffer_lookup}")
    func_name = get_func_name(tree)

    body = get_body(tree)
    if is_scalar_constant(body):
        return float(body)

    kernel = f"""
#include <metal_stdlib>
using namespace metal;

kernel void {func_name}(
    device float* out [[ buffer(0) ]],
    device float* tempA [[ buffer(1) ]],
    device float* tempB [[ buffer(2) ]],
    {buf_decls}
    uint id [[ thread_position_in_grid ]]
) {{
    {body}
}}
""".strip()
    with open("./metal_samples/kernel.metal", "w") as f:
        f.write(kernel)

    return kernel

logger = setup_logging()
buffer_lookup = {}
runtime_items = None