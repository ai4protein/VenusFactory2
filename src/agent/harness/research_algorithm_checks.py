"""Static checks for generated model interfaces; never import or execute source."""
from __future__ import annotations

import ast
import builtins
import hashlib
import symtable
import warnings

MAX_DIAGNOSTICS = 12
MAX_MESSAGE_CHARS = 600


class AlgorithmValidationError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        lines = []
        for item in diagnostics["issues"][:8]:
            location = f"line {item['line']}" if item.get("line") else ""
            if location and item.get("column"):
                location += f", column {item['column']}"
            lines.append((location+": " if location else "")+item["message"])
        super().__init__("Model source check failed: " + "\n".join(lines))


def _unimplemented(statement):
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Constant) and statement.value.value in (None, Ellipsis)
    if isinstance(statement, ast.Return):
        return statement.value is None or (
            isinstance(statement.value, ast.Constant) and statement.value.value in (None, Ellipsis))
    if isinstance(statement, ast.Raise):
        error = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
        return isinstance(error, ast.Name) and error.id == "NotImplementedError"
    return False



def algorithm_diagnostics(source):
    issues = []
    tree = None

    def issue(code, message, node=None, severity="error", line=None, column=None):
        issues.append({"code": code, "severity": severity,
                       "line": line or getattr(node, "lineno", None), "column": column, "message": message[:MAX_MESSAGE_CHARS]})

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            tree = ast.parse(source)
            # AST parsing alone permits 'return' outside a function and misses
            # compiler diagnostics such as calling a tuple after a missing comma.
            # Compilation creates code objects; it never imports/runs the source.
            compile(tree, "<research_algorithm>", "exec")
        except (SyntaxError, ValueError) as exc:
            issue("invalid_python", "Model source is not valid Python: "
                  + str(getattr(exc, "msg", exc)) + ". Repair the saved .py source before training.",
                  line=getattr(exc, "lineno", None), column=getattr(exc, "offset", None))
        for warning in recorded:
            issue("compiler_warning", str(warning.message), severity="warning", line=warning.lineno)
    if tree is not None:
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("fit", "predict"):
            node = functions.get(name)
            if (node is None or len(node.args.posonlyargs)+len(node.args.args) != 2
                    or node.args.vararg or node.args.kwarg or node.args.kwonlyargs):
                issue("invalid_entrypoint", "Implement fit(training, seed) -> fitted parameter dict or estimator; "
                      "predict(model, inputs) -> prediction dict. These must be module-level functions, "
                      "not class methods. A helper class is allowed behind them.", node)
                continue
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body.pop(0)
            if not body or all(_unimplemented(statement) for statement in body):
                issue("unfinished_entrypoint", f"{name} is an unfinished placeholder. Implement its calculations before training.", node)
        # Dynamic globals remain legal, so unresolved globals are repair hints,
        # never an admission rule. Skip modules with wildcard imports.
        if not any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in tree.body):
            try:
                symbols = symtable.symtable(source, "<research_algorithm>", "exec")
                defined = {symbol.get_name() for symbol in symbols.get_symbols()
                           if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
                defined.update(dir(builtins))
                defined.update({"__name__", "__file__", "__package__", "__doc__", "__builtins__"})
                for scope in symbols.get_children():
                    function = functions.get(scope.get_name())
                    if function is None or scope.get_name() not in {"fit", "predict"}:
                        continue
                    for symbol in scope.get_symbols():
                        name = symbol.get_name()
                        if symbol.is_global() and symbol.is_referenced() and name not in defined:
                            reference = next((n for n in ast.walk(function)
                                              if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)), function)
                            issue("unresolved_global", f"{scope.get_name()} references global {name!r} without a visible "
                                  "definition or import. Define/import it, or verify its dynamic initialization.",
                                  reference, severity="warning")
            except (SyntaxError, ValueError):
                # The compiler diagnostic above already records invalid source.
                pass
        fit = functions.get("fit")
        if fit is not None and fit.args.posonlyargs + fit.args.args:
            training_name = (fit.args.posonlyargs + fit.args.args)[0].arg
            if not any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == training_name
                       for statement in fit.body for node in ast.walk(statement)):
                issue("unused_training", "fit does not read its training argument. Fit parameters from the supplied "
                      "training['train'] rows and return the fitted model; fixed demo values or a seed alone are "
                      "not training. Existing pretrained or fixed-rule baselines use evaluate_research_model.", fit)
        if "run" in functions:
            issue("reserved_entrypoint", "The host supplies run(inputs); implement fit and predict", functions["run"])
        reserved = next((n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id.startswith("_vf_")), None)
        if reserved is not None:
            issue("reserved_name", "Names starting _vf_ are reserved for the prediction interface", reserved)
    valid = tree is not None and not any(item["severity"] == "error" for item in issues)
    # Keep blockers visible even when a source emits many compiler warnings.
    issues.sort(key=lambda item: item["severity"] != "error")
    return {"valid": valid, "issues": issues[:MAX_DIAGNOSTICS],
            "issue_count": len(issues), "omitted_issue_count": max(0, len(issues)-MAX_DIAGNOSTICS),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "algorithm_ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
                if valid else None}


def checked_algorithm_digest(diagnostics):
    if not diagnostics["valid"]:
        raise AlgorithmValidationError(diagnostics)
    return diagnostics["algorithm_ast_sha256"]


def validate_algorithm(source):
    return checked_algorithm_digest(algorithm_diagnostics(source))
