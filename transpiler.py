import inspect
import ast


def transpile(tree, source: str) -> str:
    print(ast.dump(tree, indent=4))
    return ""