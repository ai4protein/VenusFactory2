"""Carry action failures until a successful retry of the same work target."""
import re

from .contracts import digest
from .execution_outcome import EXECUTION_TOOLS, execution_error
from .tool_diagnostics import bounded_message


def _failure_message(value, workspace):
    """Keep execution evidence distinct from host scheduling guidance."""
    text = re.sub(
        r"Execution budget: [0-9]+ tool calls remain;\s*[0-9]+ read-only discovery calls remain\.\s*"
        r"Save a checkpoint when the current step is ready\.\s*"
        r"Checkpoint saves have a separate allowance:\s*[0-9]+ remain\.", "", str(value))
    lines = text.splitlines()
    evidence = [line for line in lines if not (
        line.startswith("Execution budget: ")
        and " tool calls remain; " in line
        and " read-only discovery calls remain." in line
        and "Checkpoint saves have a separate allowance:" in line
    ) and line != "Checkpoint saved. End this iteration with a short summary; do not call further tools."]
    return bounded_message("\n".join(evidence).strip() or "Tool failed", workspace, 1200)


def _name(value):
    name=str(value or "tool").rsplit("__",1)[-1]
    return "run_research_python" if name=="run_session_python" else name

def _key(name, arguments):
    name=_name(name)
    targets={
        "prepare_research_dataset":("data_file","output_file"),
        "validate_research_evaluation":("evaluation_file",),
        "train_research_model":("source_file","parent_experiment_id","variant"),
        "evaluate_research_model":("source","parent_experiment_id","variant"),
        "run_research_python":("path",),
    }
    fields=targets.get(name)
    if isinstance(arguments,dict) and fields and fields[0] in arguments:
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

def feedback_state(events, workspace, previous=None):
    failures={}
    pending=(previous or {}).get("pending")
    if isinstance(pending,list):
        failures={row["key"]:{**row,"error":_failure_message(row.get("error","Tool failed"),workspace)}
                  for row in pending if isinstance(row,dict) and row.get("key")}
    else:
        for row in (previous or {}).get("errors",[]):
            key="legacy:"+digest(row)
            failures[key]={"key":key,**row,"error":_failure_message(row.get("error","Tool failed"),workspace)}
    calls={}
    for event in events:
        identity=event.get("tool_call_id")
        if event.get("kind")=="tool_call_start":
            name=str(event.get("tool_name") or "tool")
            arguments=event.get("args") or {}
            if name.endswith("call_tool") and isinstance(arguments,dict):
                name=str(arguments.get("name") or name);arguments=arguments.get("arguments",{})
            source = _source_path(arguments.get("source_file")) if _name(name) == "train_research_model" and isinstance(arguments, dict) else None
            calls[identity]=(name[:120],_key(name,arguments),source)
        elif event.get("kind")=="tool_result":
            name,key,source=calls.get(identity,("tool","unknown:"+str(identity),None))
            semantic=execution_error(event.get("output")) if _name(name) in {_name(n) for n in EXECUTION_TOOLS} else None
            if not event.get("is_error") and not semantic:
                failures.pop(key,None)
                for old,row in list(failures.items()):
                    if old.startswith("legacy:") and _name(row["tool"])==_name(name):failures.pop(old,None)
                continue
            failures.pop(key,None)
            failures[key]={"key":key,"tool":name,
                "error":_failure_message(semantic or event.get("output") or "Tool failed",workspace)}
            if source is not None:
                failures[key]["source_file"]=source
    pending=list(failures.values())[-5:]
    return {"errors":[{"tool":row["tool"],"error":row["error"]} for row in pending],"pending":pending}

def execution_feedback(events, workspace):
    return feedback_state(events,workspace)["errors"]

def iteration_feedback(events, workspace, previous):
    return feedback_state(events,workspace,previous)["errors"]


def repair_source_context(workspace, feedback):
    """Read one current candidate implicated in an unresolved fitting failure."""
    from hashlib import sha256
    from utils.scoped_files import read_bytes
    from .session_boundary import public_value
    pending=(feedback or {}).get("pending")
    if not isinstance(pending,list):
        return None
    for row in reversed(pending[-5:]):
        if not isinstance(row,dict) or _name(row.get("tool")) != "train_research_model":
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
        return {"path":path,"sha256":sha256(raw).hexdigest(),"code":code,
                "redacted":code!=original,"error":_failure_message(row.get("error","Fitting failed"),workspace),
                "instruction":"This is the current saved source for an unresolved fitting action. "
                    "Repair its complete implementation using the observed error and model interface, save it, and fit again. "
                    "For sequence-regression encoding errors, get_research_baseline(reference, output_file) can save a verified comparison implementation in a new file. "
                    "The file may have changed since the failure; do not replace working code with placeholders."}
    return None
