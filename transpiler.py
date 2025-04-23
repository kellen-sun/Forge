import inspect
import ast
import logging

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

def get_expr(node):
    if isinstance(node, ast.BinOp):
        left = get_expr(node.left)
        right = get_expr(node.right)
        op = node.op
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
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
                    out.append(f"out[id] = {get_expr(stmt.value)};")
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

def transpile(tree, source: str) -> str:
    logger.info(f"AST:\n{ast.dump(tree, indent=4)}")

    args = get_args(tree)
    buf_decls = "\n    ".join([
        f"device float* {arg} [[ buffer({i}) ]]," for i, arg in enumerate(args)
    ])
    func_name = get_func_name(tree)

    body = get_body(tree)

    kernel = f"""
#include <metal_stdlib>
using namespace metal;

kernel void {func_name}(
    {buf_decls}
    device float* out [[ buffer({len(args)}) ]],
    uint id [[ thread_position_in_grid ]]
) {{
    {body}
}}
""".strip()
    with open("./metal_samples/kernel.metal", "w") as f:
        f.write(kernel)

    return kernel

logger = setup_logging()