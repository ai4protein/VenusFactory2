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
            if item.get("source_excerpt"):
                lines.append(item["source_excerpt"])
                lines.append(" " * (item["excerpt_column"] - 1) + "^")
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



def _entrypoint_message(name, node, tree):
    signature = "fit(training, seed)" if name == "fit" else "predict(model, inputs)"
    if node is None:
        methods = [(owner.name, method) for owner in tree.body if isinstance(owner, ast.ClassDef)
                   for method in owner.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and method.name == name]
        if methods:
            owner, method = methods[0]
            detail = (f"Found {owner}.{name} as a class method, not a module entrypoint. "
                      f"Place def {signature} at file scope, outside the class, without self or cls. ")
            location = method
        else:
            detail = f"The module has no callable {name} entrypoint. "
            location = next((n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name), None)
    else:
        args = [a.arg for a in node.args.posonlyargs + node.args.args]
        detail = f"{name} declares parameters {args!r}; the host passes exactly two positional arguments. "
        location = node
    if name == "fit":
        repair = ("It may call a helper class, but must return the fitted estimator or one dict containing "
                  "the learner and its fitted transformations. Read rows from training['train'].")
    else:
        repair = ("Recover the learner and fitted transformations from the single model argument. "
                  "inputs is one query dict. Return a dict with the declared prediction key (scalar or multilabel probabilities); do not refit on query targets.")
    return f"Implement module-level {signature}. " + detail + repair, location


def _input_contract_hints(functions, input_fields, issue):
    """Conservative AST hints, not an algorithm whitelist or a static type proof."""
    fields = set(input_fields)
    if not fields or not all(isinstance(name, str) for name in fields):
        return
    fit = functions.get("fit")
    if fit and fit.args.posonlyargs + fit.args.args:
        training = (fit.args.posonlyargs + fit.args.args)[0].arg
        for node in ast.walk(fit):
            if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)) or len(node.generators) != 1:
                continue
            generator = node.generators[0]
            iterator = generator.iter
            if not (isinstance(generator.target, ast.Name) and isinstance(iterator, ast.Subscript)
                    and isinstance(iterator.value, ast.Name) and iterator.value.id == training
                    and isinstance(iterator.slice, ast.Constant) and iterator.slice.value == "train"):
                continue
            row = generator.target.id
            for access in ast.walk(node.elt):
                if (isinstance(access, ast.Subscript) and isinstance(access.value, ast.Name)
                        and access.value.id == row and isinstance(access.slice, ast.Constant)
                        and isinstance(access.slice.value, str) and access.slice.value in fields
                        and access.slice.value not in {"id", "group", "task_id", "inputs", "target"}):
                    key = repr(access.slice.value)[:100]
                    issue("training_row_field", f"Training rows store the declared feature at {row}['inputs'][{key}], "
                          f"not directly at {row}[{key}]. Targets are {row}['target']. "
                          "Check this indexing unless your code explicitly transformed the row first.",
                          access, severity="warning")
    predict = functions.get("predict")
    if predict and len(predict.args.posonlyargs + predict.args.args) == 2 and "inputs" not in fields:
        query = (predict.args.posonlyargs + predict.args.args)[1].arg
        for node in ast.walk(predict):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == query
                    and isinstance(node.slice, ast.Constant) and node.slice.value == "inputs"):
                issue("prediction_input_nesting", f"{query} is already the query feature dictionary. "
                      f"Read {query}[field] directly; the prepared contract has no outer 'inputs' key. "
                      "Check this indexing unless your code intentionally wrapped the query.", node, severity="warning")


def algorithm_diagnostics(source, *, input_fields=()):
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
            line, column = getattr(exc, "lineno", None), getattr(exc, "offset", None)
            source_lines = source.splitlines()
            if isinstance(line, int) and 1 <= line <= len(source_lines):
                text = source_lines[line - 1]
                position = max(0, min((column or 1) - 1, len(text)))
                start = max(0, position - 100)
                issues[-1].update(source_excerpt=text[start:start+200], excerpt_column=position-start+1)
        for warning in recorded:
            issue("compiler_warning", str(warning.message), severity="warning", line=warning.lineno)
    if tree is not None:
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("fit", "predict"):
            node = functions.get(name)
            if (node is None or len(node.args.posonlyargs)+len(node.args.args) != 2
                    or node.args.vararg or node.args.kwarg or node.args.kwonlyargs):
                message, location = _entrypoint_message(name, node, tree)
                issue("invalid_entrypoint", message, location)
                continue
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body.pop(0)
            if not body or all(_unimplemented(statement) for statement in body):
                issue("unfinished_entrypoint", f"{name} is an unfinished placeholder. Implement its calculations before training.", node)
        _input_contract_hints(functions, input_fields, issue)
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
