"""Reproducible group splits from user-selected experimental table columns."""
from __future__ import annotations
import csv
import hashlib
import io
import math
from collections import defaultdict

from utils.scoped_files import read_bytes
from .research_development import require, immutable_copy, inspect_evaluation
from .research_evaluation import validate_data, evaluation_bytes
from .research_episodes import prepare_episodes
from .research_input_profile import training_input_profile



def model_interface(task, columns, *, episodic=False):
    """Describe the callable shape without observations or fitted-method choices."""
    fields = {name: "<original CSV string>" for name in columns[:4]}
    result = {
        "schema": "venusfactory.training-interface.v1",
        "entrypoints": "Module-level fit and predict functions; helper classes may be used behind them.",
        "shape_example": {
            "kind": "placeholder values only; not observations",
            "training_row": {"inputs": fields, "target": "<measured target>"},
            "prediction_inputs": fields,
            "omitted_input_fields": max(0, len(columns)-len(fields)),
        },
        "input_fields": {name: "original CSV string" for name in columns},
        "fit": {
            "signature": "fit(training, seed)",
            "samples": "training['train'] is a list; each row contains an inputs dict and target.",
            "features": "Encode inputs into features. Numeric estimators cannot fit raw dictionaries or amino-acid strings.",
            "preprocessing": "Learn transformations from training rows only; retain fitted transformers with the learner.",
            "returns": "Parameters learned from the supplied training rows: a fitted estimator or dict with fitted transformers; not fixed demo values.",
        },
        "predict": {
            "signature": "predict(model, inputs)",
            "inputs": "ONE input_fields dictionary; this is not a list of rows.",
            "features": "Apply the same feature transformation used during fitting.",
            "returns": {"value": "finite number" if task == "regression" else "scalar class label"},
            "output_key": "Use prediction_key from train_research_model; its default is value.",
        },
    }
    if episodic:
        support = [{"id": "<support observation ID>", "inputs": dict(fields), "target": "<support measurement>"}]
        result["shape_example"]["training_row"]["task_id"] = "<training task ID>"
        result["shape_example"]["training_row"]["inputs"] = {**fields, "support": support}
        result["shape_example"]["prediction_inputs"] = {**fields, "support": support}
        result["input_fields"]["support"] = "List of same-task {id, inputs, target} support observations. Never include the query target."
        result["fit"]["tasks"] = "Group training rows by task_id. Each query carries its fixed support observations in inputs['support']."
    return result



def protected_data_access(data):
    """Describe the training view separately from the opaque evaluation handle."""
    preparation = data["preparation"]
    source = preparation["source_file"]
    tabular = source.lower().endswith((".tsv", ".tab"))
    result = {
        "training_data": {
            "file": source, "format": "tsv" if tabular else "csv",
            "delimiter": "\t" if tabular else ",", "scope": "training_only",
            "rows": len(data["splits"]["train"]),
            "target_column": preparation["target_column"],
            "input_columns": preparation["input_columns"],
        },
        "evaluation_access": "host_reference",
        "data_usage": "Read training_data.file as a table for analysis. Pass evaluation_file to training/review tools; it is an opaque reference, not sample data.",
    }
    if "unlabeled" in preparation:
        pool = preparation["unlabeled"]
        tabular_pool = pool["source_file"].lower().endswith((".tsv", ".tab"))
        result["unlabeled_data"] = {
            "file": pool["source_file"], "scope": "unlabeled_only", "rows": len(data["unlabeled"]),
            "format": "tsv" if tabular_pool else "csv", "delimiter": "\t" if tabular_pool else ",",
            "input_columns": preparation["input_columns"], "id_column": pool["id_column"],
        }
    if "support" in data:
        result["training_data"]["rows"] += len(data["support"]["train"])
        result["training_data"]["scope"] = "training_tasks_only"
        result["training_data"]["task_column"] = preparation["task_column"]
    return result


def prepared_details(data):
    """One metadata view for the initial result and checkpoint restoration."""
    preparation = data["preparation"]
    result = {
        "model_interface": model_interface(data["task"], preparation["input_columns"], episodic="support" in data),
        "training_profile": training_input_profile(data["splits"]["train"], preparation["input_columns"]),
    }
    if "support" in data:
        result["episodes"] = {
            "task_column": preparation["task_column"], "support_size": preparation["episode"]["support_size"],
            "query_support_field": "support", "partitions": {
                name: {"tasks": len({row["task_id"] for row in queries}), "query_rows": len(queries),
                       "support_rows": len(data["support"][name])}
                for name, queries in data["splits"].items()},
            "access": "Fitting receives only training tasks. Each held-out prediction receives its task's support, never its query target.",
        }
    if "unlabeled" in preparation:
        result["unlabeled_pool"] = {**preparation["unlabeled"], "rows": len(data["unlabeled"])}
    return result


def read_table(workspace, path, *, data_store=None, maximum=21024):
    require(isinstance(path, str) and path.lower().endswith((".csv", ".tsv", ".tab")),
            "Dataset preparation currently accepts CSV/TSV tables; export the measured table to CSV/TSV before uploading")
    raw = data_store.original_bytes(path) if data_store else read_bytes(workspace, path, max_bytes=20*1024**2)
    delimiter = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter)
    fields = reader.fieldnames
    require(fields and all(fields) and len(fields) == len(set(fields)), "Use a table with distinct nonempty column names")
    rows = []
    for index, row in enumerate(reader, 1):
        require(index <= maximum, f"At most {maximum:,} table rows are supported; select a documented subset first")
        require(None not in row and all(value is not None for value in row.values()), f"Malformed table row {index}")
        rows.append(row)
    return hashlib.sha256(raw).hexdigest(), fields, rows



def prepare_dataset(workspace, *, data_file, target_column, input_columns, group_column,
                    provenance, grouping, id_column="", task="regression", seed=0,
                    validation_fraction=0.15, test_fraction=0.15, output_file="", data_store=None,
                    task_column="", support_size=0, unlabeled_file="", unlabeled_id_column=""):
    require(task in {"regression","classification"}, "Declare regression or classification")
    require(type(seed) is int and 0 <= seed < 2**32, "Use an unsigned 32-bit split seed")
    require(all(type(x) in (int,float) and math.isfinite(x) and 0 < x < 1
                for x in (validation_fraction,test_fraction))
            and validation_fraction+test_fraction < 0.8, "Holdout fractions must be positive and sum to less than 0.8")
    require(isinstance(input_columns,list) and 1 <= len(input_columns) <= 128
            and all(isinstance(x,str) and x for x in input_columns)
            and len(set(input_columns))==len(input_columns), "Select distinct input columns")
    require(target_column not in input_columns, "The prediction target cannot be an input feature")
    require(all(isinstance(x,str) and x.strip() for x in (target_column,group_column,provenance,grouping)),
            "Declare the measured target, biological grouping and provenance")
    require(all(isinstance(value, str) for value in (task_column, unlabeled_file, unlabeled_id_column)),
            "Task and unlabeled selectors must be column names or empty strings")
    require(type(support_size) is int and (1 <= support_size <= 64 if task_column else support_size == 0),
            "Set task_column and support_size=1..64 together for task episodes")
    require(not task_column or "support" not in input_columns, "support is reserved for episode observations")
    require(not unlabeled_id_column or unlabeled_file, "unlabeled_id_column requires unlabeled_file")
    source_sha, fields, source_rows = read_table(workspace, data_file, data_store=data_store)
    requested=[target_column,group_column,*input_columns]+([id_column] if id_column else [])+([task_column] if task_column else [])
    require(set(requested)<=set(fields), "Selected columns are missing; available columns: "+", ".join(fields)[:1000])
    rows=[];groups=defaultdict(list)
    for index,row in enumerate(source_rows,1):
        identity=row[id_column].strip() if id_column else source_sha[:12]+":"+str(index)
        group=row[group_column].strip()
        target=row[target_column].strip()
        require(identity and group and target, f"Missing identity, group or measured target at row {index}")
        if task=="regression":
            try: target=float(target)
            except ValueError as exc:raise ValueError(f"Non-numeric measured target at row {index}") from exc
            require(math.isfinite(target), f"Nonfinite measured target at row {index}")
        record={"id":identity,"group":group,"inputs":{col:row[col] for col in input_columns},"target":target}
        if task_column:
            record["task_id"] = row[task_column].strip()
            require(bool(record["task_id"]), f"Missing task identity at row {index}")
        rows.append(record);groups[group].append(record)
    require(len(groups)>=3, "At least three biological groups are required for disjoint splits")
    ordered=sorted(groups,key=lambda g:(hashlib.sha256((str(seed)+"\\0"+g).encode()).hexdigest(),g))
    nv=max(1,round(len(ordered)*validation_fraction));nt=max(1,round(len(ordered)*test_fraction))
    require(nv+nt<len(ordered), "Holdout fractions leave no training groups")
    assignment={"validation":ordered[:nv],"test":ordered[nv:nv+nt],"train":ordered[nv+nt:]}
    data={"task":task,"provenance":provenance,"grouping":grouping,
          "splits":{name:[row for group in selected for row in groups[group]] for name,selected in assignment.items()},
          "preparation":{"source_file":data_file,"source_sha256":source_sha,
                         "target_column":target_column,"input_columns":input_columns,
                         "id_column":id_column,"group_column":group_column,"seed":seed,
                         "validation_fraction":validation_fraction,"test_fraction":test_fraction,
                         "method":"sha256_ordered_whole_groups","source_rows":len(rows)}}
    if task_column:
        splits, support, episode = prepare_episodes(rows, seed=seed, support_size=support_size,
            validation_fraction=validation_fraction, test_fraction=test_fraction)
        data["splits"] = splits
        data["support"] = support
        data["preparation"].update(task_column=task_column, episode=episode,
                                   method="sha256_connected_task_groups_support_query")
    if unlabeled_file:
        pool_sha, pool_fields, pool_rows = read_table(workspace, unlabeled_file, data_store=data_store, maximum=20000)
        require(target_column not in pool_fields, "The unlabeled table must not contain the measured target column")
        selected = input_columns+([unlabeled_id_column] if unlabeled_id_column else [])
        require(set(selected) <= set(pool_fields), "The unlabeled table must contain the selected input and ID columns")
        data["unlabeled"] = [
            {"id": row[unlabeled_id_column].strip() if unlabeled_id_column else pool_sha[:12]+":"+str(index),
             "inputs": {column: row[column] for column in input_columns}}
            for index, row in enumerate(pool_rows, 1)]
        require(bool(data["unlabeled"]), "The unlabeled table contains no observations")
        data["preparation"]["unlabeled"] = {"source_file": unlabeled_file, "source_sha256": pool_sha,
                                                "id_column": unlabeled_id_column}
    validate_data(data,"meta_learning" if task_column else "supervised")
    content=evaluation_bytes(data)
    identity=hashlib.sha256(content).hexdigest()
    output_file=output_file or "data/evaluation-"+identity[:16]+".json"
    if data_store:
        data_store.expose_training(data_file,data)
        data_store.write_evaluation(output_file,content)
    else:
        immutable_copy(workspace,output_file,content)
    result=inspect_evaluation(workspace,output_file,strategy="meta_learning" if task_column else "supervised",data_store=data_store)
    result.update(source_file=data_file,table_sha256=source_sha,source_rows=len(rows),
                  input_columns=input_columns,input_type="Original table strings; convert numerical features in fit/predict.",
                  **prepared_details(data),
                  next_step="Write a complete fit(training, seed) and predict(model, inputs) module, then call train_research_model with this evaluation_file and variant=baseline.")
    if data_store:
        result.update(protected_data_access(data))
    return result
