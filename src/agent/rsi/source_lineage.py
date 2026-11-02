"""Source identity for lineage admission, not a test of algorithmic novelty."""

import ast
import hashlib


def program_hash(source: str) -> str:
    """Ignore formatting, comments and docstrings; preserve executable syntax.

    Different hashes are necessary but insufficient evidence of useful algorithmic
    change. Independent fit/predict replay and development improvement are separate
    gates. Renaming variables can still change this hash.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
