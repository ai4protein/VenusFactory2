"""Carry action failures until a successful retry of the same work target."""
import re

from .contracts import digest
from .execution_outcome import EXECUTION_TOOLS, execution_error
from .tool_diagnostics import bounded_message


def _failure_message(value, workspace):
    """Keep execution evidence distinct from exact host scheduling footers."""
    text = str(value)
    normal = "Save a checkpoint when the current step is ready."
    urgent = "Save submit_research_checkpoint now with actual artifacts and only completed steps."
    # Native transports can concatenate separate MCP text blocks without a
    # newline. Match only the complete terminal footer emitted by the host;
    # a partial or merely similar sentence remains part of the diagnostic.
    footer = re.search(
        r"Execution budget: (0|[1-9][0-9]*) tool calls remain; "
        r"(?:0|[1-9][0-9]*) read-only discovery calls remain\. "
        + "(" + re.escape(normal) + "|" + re.escape(urgent) + ")"
        + r" Checkpoint saves have a separate allowance: (?:0|[1-9][0-9]*) remain\.\Z",
        text,
    )
    if footer and footer.group(2) == (urgent if int(footer.group(1)) <= 2 else normal):
        text = text[:footer.start()]
    evidence = [line for line in text.splitlines()
                if line != "Checkpoint saved. End this iteration with a short summary; do not call further tools."]
    return bounded_message("\n".join(evidence).strip() or "Tool failed", workspace, 1200)



def failure_summary(value):
    """Keep the diagnostic headline, not a source excerpt or caret underline."""
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not lines:
        return "Tool failed"
    # A Python traceback puts its exception after the stack; typed tool errors
    # and source-check diagnostics put their actual message first.
    runner_traceback = re.fullmatch(
        r"Execution exited with code (?:0|-?[1-9][0-9]*): Traceback \(most recent call last\):", lines[0])
    if lines[0].startswith("Traceback (most recent call last):") or runner_traceback:
        lines = [line for line in lines if not re.fullmatch(r"[~^]+", line)]
        return lines[-1][:700]
    return lines[0][:700]


def _name(value):
    name=str(value or "tool").rsplit("__",1)[-1]
    return {"run_session_python":"run_research_python", "edit_session_file":"write_session_file"}.get(name,name)

def _key(name, arguments):
    name=_name(name)
    targets={
        "write_session_file":("path",),
        "prepare_research_dataset":("data_file","output_file"),
        "validate_research_evaluation":("evaluation_file",),
        "train_research_model":("source_file","parent_experiment_id","variant"),
        "evaluate_research_model":("source","parent_experiment_id","variant"),
        "run_research_python":("path",),
    }
    fields=targets.get(name)
    if name == "submit_research_checkpoint":
        # Each successful save replaces this workspace's active checkpoint.
        # Plan/evidence validation still decides which steps it can complete.
        value = None
    elif isinstance(arguments,dict) and fields and fields[0] in arguments:
        value={key:arguments.get(key,"auto" if key=="variant" else "") for key in fields}
        if "variant" in value and value["variant"] == "auto":
            value["variant"] = "candidate" if value["parent_experiment_id"] else "baseline"
    else:value=arguments
    return digest([name,value])

def _source_path(value):
    if not isinstance(value,str) or not 0<len(value)<=512 or not value.endswith(".py"):
        return None
    if value.startswith("/") or chr(92) in value or ":" in value:
        return None
    if any(not part or part in {".",".."} or part.startswith(".") for part in value.split("/")):
        return None
    return value

def _request_key(name, arguments):
    """Bind requests to the actual tool signature, including optional defaults."""
    from inspect import signature

    from .research_workspace import ResearchWorkspace

    try:
        bound = signature(ResearchWorkspace.train_research_model).bind_partial(None, **arguments)
        bound.apply_defaults()
        value = dict(bound.arguments)
        value.pop("self", None)
        if value.get("variant") == "auto":
            value["variant"] = "candidate" if value.get("parent_experiment_id") else "baseline"
    except TypeError:
        # Invalid/missing arguments cannot borrow the identity of a valid call.
        value = {"unbound_arguments": arguments}
    return "request:" + digest([_name(name), value])


def detached_request_failure(row, workspace):
    """Preserve legacy attached rejections without inventing their request scope."""
    failure = row.get("request_failure")
    if not isinstance(failure, dict):
        return None
    return {"key": "legacy-request:" + digest([row.get("key"), failure]),
            "tool": row["tool"], "failure_kind": "request", "action_key": row.get("key"),
            "error": _failure_message(failure.get("error", "Tool failed"), workspace)}


def _request_lifecycle(failures, previous, workspace, episode_index):
    """Archive earlier unexecuted requests only when the host advances a cycle."""
    previous = previous or {}
    prior = previous.get("episode_index")
    for value in (prior, episode_index):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Host episode index must be a nonnegative integer")
    if episode_index is not None and prior is not None and episode_index < prior:
        raise ValueError("Host episode index cannot move backwards")
    history = {}
    for row in previous.get("request_history", []):
        # Only host-produced, explicitly unexecuted records enter this history.
        if not isinstance(row, dict) or row.get("disposition") != "not_executed":
            raise ValueError("Invalid request history")
        for field in ("episode_index", "archived_at_episode"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError("Invalid request history episode")
        if row["archived_at_episode"] <= row["episode_index"]:
            raise ValueError("Request history must precede its archival episode")
        current = episode_index if episode_index is not None else prior
        if current is not None and row["archived_at_episode"] > current:
            raise ValueError("Request history cannot come from a future episode")
        identity = (row["key"], row["episode_index"])
        history[identity] = {**row, "error": _failure_message(row.get("error", "Tool failed"), workspace)[:600]}
    for key, row in list(failures.items()):
        if row.get("failure_kind") != "request" or episode_index is None:
            continue
        created = row.get("episode_index")
        if created is None:
            # Legacy requests have no proven age. Retain them for this cycle.
            row["episode_index"] = episode_index
            continue
        if type(created) is not int or created < 0 or created > episode_index:
            raise ValueError("Invalid request episode")
        if created < episode_index:
            record = {"key": key, "tool": row["tool"], "episode_index": created,
                      "archived_at_episode": episode_index, "disposition": "not_executed",
                      "error": _failure_message(row["error"], workspace)[:600]}
            history[(key, created)] = record
            failures.pop(key)
    rows = list(history.values())
    omitted = previous.get("request_history_omitted", 0)
    if type(omitted) is not int or omitted < 0:
        raise ValueError("Invalid request history count")
    result = {}
    if episode_index is not None or prior is not None:
        result["episode_index"] = episode_index if episode_index is not None else prior
    if rows:
        result["request_history"] = rows[-8:]
    omitted += max(0, len(rows) - 8)
    if omitted:
        result["request_history_omitted"] = omitted
    return result


def feedback_state(events, workspace, previous=None, *, development=None, episode_index=None):
    failures = {}
    pending = (previous or {}).get("pending")
    if isinstance(pending, list):
        for row in pending:
            if not isinstance(row, dict) or not row.get("key"):
                continue
            saved = {**row, "error": _failure_message(row.get("error", "Tool failed"), workspace)}
            saved.pop("request_failure", None)
            if saved.get("failure_kind") == "request":
                saved.pop("training_attempt_id", None)
            failures[saved["key"]] = saved
            request = detached_request_failure(row, workspace)
            if request is not None:
                failures[request["key"]] = request
    else:
        for row in (previous or {}).get("errors", []):
            key = "legacy:" + digest(row)
            failures[key] = {"key": key, **row, "error": _failure_message(row.get("error", "Tool failed"), workspace)}
    lifecycle = _request_lifecycle(failures, previous, workspace, episode_index)
    calls = {}
    for event in events:
        identity = event.get("tool_call_id")
        if event.get("kind") == "tool_call_start":
            name = str(event.get("tool_name") or "tool")
            arguments = event.get("args") or {}
            if name.endswith("call_tool") and isinstance(arguments, dict):
                name = str(arguments.get("name") or name)
                arguments = arguments.get("arguments", {})
            training = _name(name) == "train_research_model"
            source = _source_path(arguments.get("source_file")) if training and isinstance(arguments, dict) else None
            request_key = _request_key(name, arguments) if training else None
            calls[identity] = (name[:120], _key(name, arguments), source, request_key)
        elif event.get("kind") == "tool_result":
            name, key, source, request_key = calls.get(identity, ("tool", "unknown:"+str(identity), None, None))
            semantic = execution_error(event.get("output")) if _name(name) in {_name(n) for n in EXECUTION_TOOLS} else None
            if not event.get("is_error") and not semantic:
                failures.pop(key, None)
                failures.pop(request_key, None)
                for old, row in list(failures.items()):
                    if old.startswith("legacy:") and _name(row["tool"]) == _name(name):
                        failures.pop(old, None)
                continue
            failure = {"key": key, "tool": name,
                "error": _failure_message(semantic or event.get("output") or "Tool failed", workspace)}
            if source is not None:
                failure["source_file"] = source
            if request_key is not None:
                from .research_attempt_resolution import attempt_id
                identity = attempt_id(event.get("output"))
                if identity:
                    failure["training_attempt_id"] = identity
                    failures.pop(request_key, None)
                else:
                    # No receipt means no new fitting evidence. Only a later
                    # matching request result can resolve this separate failure.
                    failure.update(key=request_key, action_key=key, failure_kind="request")
                    if episode_index is not None:
                        failure["episode_index"] = episode_index
                    key = request_key
            failures.pop(key, None)
            failures[key] = failure
    rows = list(failures.values())
    # Repeated request denials cannot evict observed execution evidence.
    pending = ([row for row in rows if row.get("failure_kind") != "request"][-5:]
               + [row for row in rows if row.get("failure_kind") == "request"][-5:])
    result = {"pending": pending}
    if (previous or {}).get("superseded"):
        result["superseded"] = list(previous["superseded"])
    if development is not None and any(row.get("training_attempt_id") for row in pending):
        from .research_attempt_resolution import resolve_attempts
        result = resolve_attempts(result, development)
    result["errors"] = [{"tool": row["tool"], "error": row["error"]} for row in result["pending"]]
    result.update(lifecycle)
    return result

def execution_feedback(events, workspace):
    return feedback_state(events,workspace)["errors"]

def iteration_feedback(events, workspace, previous, *, development=None):
    return feedback_state(events,workspace,previous,development=development)["errors"]


def repair_source_context(workspace, feedback, *, input_fields=(), task=""):
    """Read one current candidate implicated in an unresolved fitting failure."""
    from hashlib import sha256
    from utils.scoped_files import read_bytes
    from .session_boundary import public_value
    from .research_algorithm_checks import algorithm_diagnostics
    pending=(feedback or {}).get("pending")
    if not isinstance(pending,list):
        return None
    execution = [row for row in pending if isinstance(row, dict) and row.get("failure_kind") != "request"]
    for row in reversed(execution[-5:]):
        if _name(row.get("tool")) != "train_research_model":
            continue
        path=_source_path(row.get("source_file"))
        if path is None:
            continue
        try:
            raw=read_bytes(workspace,path,max_bytes=12000)
            original=raw.decode("utf-8")
        except (OSError,ValueError,UnicodeError):
            continue
        code=public_value(original,workspace)
        checks=algorithm_diagnostics(original, input_fields=input_fields)
        reference_hint = ("For sequence regression or multilabel tasks, get_research_baseline(reference, output_file, task) "
                          "can save a verified comparison implementation in a new file. ")
        if task == "multilabel":
            reference_hint = ("For a sequence-only multilabel comparison, inspect get_research_baseline(task='multilabel'): "
                              "training_prevalence, aac_logistic and terminal_logistic are available fixed controls. "
                              "The parsed row['target'] maps label names directly to 0/1/None; the CSV target-column name is not a nested key. ")
        result = {"path":path,"sha256":sha256(raw).hexdigest(),"code":code,
                "redacted":code!=original,"source_check":public_value(checks,workspace),"error":_failure_message(row.get("error","Fitting failed"),workspace),
                "instruction":"This is the current saved source for an unresolved fitting action. "
                    "Repair the observed failure using edit_session_file(path, old_text, new_text, expected_sha256) for a focused change. "
                    "Use this source SHA256, preserve working code, then fit again. "
                    + reference_hint + "The file may have changed since the failure; do not replace working code with placeholders."}
        requests = [item for item in pending if isinstance(item, dict)
                    and item.get("failure_kind") == "request" and item.get("action_key") == row.get("key")]
        if requests:
            result["request_failures"] = [{"tool": item["tool"],
                "error": _failure_message(item.get("error", "Tool failed"), workspace)} for item in requests[-5:]]
        return result
    return None
