"""Reproducible group splits from user-selected experimental table columns."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from collections import defaultdict

from utils.scoped_files import read_bytes
from .research_development import require, immutable_copy, inspect_evaluation
from .research_evaluation import validate_data
from .research_input_profile import training_input_profile



def model_interface(task, columns):
    """Describe the callable shape without observations or fitted-method choices."""
    fields = {name: "<original CSV string>" for name in columns[:4]}
    return {
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


def protected_data_access(data):
    """Describe the training view separately from the opaque evaluation handle."""
    preparation = data["preparation"]
    source = preparation["source_file"]
    tabular = source.lower().endswith((".tsv", ".tab"))
    return {
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


def prepare_dataset(workspace, *, data_file, target_column, input_columns, group_column,
                    provenance, grouping, id_column="", task="regression", seed=0,
                    validation_fraction=0.15, test_fraction=0.15, output_file="", data_store=None):
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
    require(isinstance(data_file, str) and data_file.lower().endswith((".csv", ".tsv", ".tab")),
            "Dataset preparation currently accepts CSV/TSV tables; export the measured table to CSV/TSV before uploading")
    raw=data_store.original_bytes(data_file) if data_store else read_bytes(workspace,data_file,max_bytes=20*1024**2)
    source_sha=hashlib.sha256(raw).hexdigest()
    text=raw.decode("utf-8-sig")
    delimiter="\t" if data_file.lower().endswith((".tsv",".tab")) else ","
    reader=csv.DictReader(io.StringIO(text),delimiter=delimiter)
    fields=reader.fieldnames
    require(fields and all(fields) and len(fields)==len(set(fields)), "Use a table with distinct nonempty column names")
    requested=[target_column,group_column,*input_columns]+([id_column] if id_column else [])
    require(set(requested)<=set(fields), "Selected columns are missing; available columns: "+", ".join(fields)[:1000])
    rows=[];groups=defaultdict(list)
    for index,row in enumerate(reader,1):
        require(index<=21024, "At most 21,024 table rows are supported; select a documented subset first")
        require(None not in row and all(v is not None for v in row.values()), f"Malformed table row {index}")
        identity=row[id_column].strip() if id_column else source_sha[:12]+":"+str(index)
        group=row[group_column].strip()
        target=row[target_column].strip()
        require(identity and group and target, f"Missing identity, group or measured target at row {index}")
        if task=="regression":
            try: target=float(target)
            except ValueError as exc:raise ValueError(f"Non-numeric measured target at row {index}") from exc
            require(math.isfinite(target), f"Nonfinite measured target at row {index}")
        record={"id":identity,"group":group,"inputs":{col:row[col] for col in input_columns},"target":target}
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
    validate_data(data,"supervised")
    content=json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    identity=hashlib.sha256(content).hexdigest()
    output_file=output_file or "data/evaluation-"+identity[:16]+".json"
    if data_store:
        data_store.expose_training(data_file,data)
        data_store.write_evaluation(output_file,content)
    else:
        immutable_copy(workspace,output_file,content)
    result=inspect_evaluation(workspace,output_file,data_store=data_store)
    result.update(source_file=data_file,table_sha256=source_sha,source_rows=len(rows),
                  input_columns=input_columns,input_type="Original table strings; convert numerical features in fit/predict.",
                  model_interface=model_interface(task,input_columns),
                  training_profile=training_input_profile(data["splits"]["train"],input_columns),
                  next_step="Write a complete fit(training, seed) and predict(model, inputs) module, then call train_research_model with this evaluation_file and variant=baseline.")
    if data_store:
        result.update(protected_data_access(data))
    return result
