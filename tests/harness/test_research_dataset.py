"""Dataset preparation preserves measurements and group isolation."""
import csv
import io
import json
import pytest
from agent.harness.research_dataset import prepare_dataset

def table(path,sep=","):
    stream=io.StringIO()
    writer=csv.writer(stream,delimiter=sep)
    writer.writerow(["id","group","x","target"])
    for n in range(60):writer.writerow([str(n),"g"+str(n//2),str(n),str(2*n+1)])
    path.write_text(stream.getvalue())

def args():
    return dict(data_file="data.csv",target_column="target",input_columns=["x"],group_column="group",
                id_column="id",provenance="Synthetic software fixture",grouping="Pairs of related observations",seed=42)

@pytest.mark.parametrize("sep,suffix",[(",","csv"),("\t","tsv")])
def test_split_is_reproducible_complete_and_group_disjoint(tmp_path,sep,suffix):
    path=tmp_path/("data."+suffix);table(path,sep);kw=args();kw["data_file"]=path.name
    first=prepare_dataset(tmp_path,**kw);second=prepare_dataset(tmp_path,**kw)
    assert first["evaluation_file"]==second["evaluation_file"] and first["source_rows"]==60
    assert "target" not in first and "splits" in first
    data=json.loads((tmp_path/first["evaluation_file"]).read_text())
    groups=[{r["group"] for r in rows} for rows in data["splits"].values()]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    rows=[row for split in data["splits"].values() for row in split]
    assert len(rows)==60 and len({row["id"] for row in rows})==60
    assert all(row["target"]==2*int(row["id"])+1 and row["inputs"]=={"x":row["id"]} for row in rows)
    changed=dict(kw,seed=43)
    assert prepare_dataset(tmp_path,**changed)["evaluation_file"]!=first["evaluation_file"]

@pytest.mark.parametrize("change,match",[
    ({"input_columns":["target"]},"target cannot"),
    ({"target_column":"invented"},"missing"),
    ({"validation_fraction":0.5,"test_fraction":0.4},"fractions"),
    ({"seed":True},"seed"),
    ({"data_file":"../outside.csv"},None)])
def test_invalid_selection_does_not_create_evaluation(tmp_path,change,match):
    table(tmp_path/"data.csv");kw={**args(),**change}
    with pytest.raises((ValueError,FileNotFoundError),match=match):prepare_dataset(tmp_path,**kw)
    assert not (tmp_path/"data").exists()

@pytest.mark.parametrize("content,match",[
    ("id,group,x,target\n1,a,1,nan\n","Nonfinite"),
    ("id,group,x,target\n1,a,1,not-a-number\n","Non-numeric"),
    ("id,group,x,target\n1,a,1,2,extra\n","Malformed"),
    ("id,group,x,target,target\n1,a,1,2,3\n","distinct"),
    ("id,group,x,target\n1,a,1,2\n2,b,2,3\n","three biological")])
def test_invalid_measurements_are_rejected(tmp_path,content,match):
    (tmp_path/"data.csv").write_text(content)
    with pytest.raises(ValueError,match=match):prepare_dataset(tmp_path,**args())
    assert not (tmp_path/"data").exists()


@pytest.mark.asyncio
async def test_table_to_fitted_baseline_and_measured_revision(tmp_path):
    from agent.harness.research_workspace import ResearchWorkspace
    service=ResearchWorkspace(tmp_path/"work",tmp_path/"control",{"steps":[]},development_root=tmp_path/"development")
    table(service.workspace/"data.csv")
    prepared=await service.prepare_research_dataset(**args())
    assert service.receipts[-1]["tool"]=="prepare_research_dataset"
    baseline="def fit(training, seed):\n    return {'mean':sum(r['target'] for r in training['train'])/len(training['train'])}\ndef predict(model, inputs):\n    return {'value':model['mean']}\n"
    await service.write_research_file("baseline.py",baseline)
    first=await service.train_research_model("Reference","baseline.py",prepared["evaluation_file"],"Training mean",variant="baseline")
    revision="def fit(training, seed):\n    rows=training['train'];x=[float(r['inputs']['x']) for r in rows];y=[r['target'] for r in rows]\n    mx=sum(x)/len(x);my=sum(y)/len(y)\n    a=sum((u-mx)*(v-my) for u,v in zip(x,y))/sum((u-mx)**2 for u in x)\n    return {'a':a,'b':my-a*mx}\ndef predict(model, inputs):\n    return {'value':model['a']*float(inputs['x'])+model['b']}\n"
    await service.write_research_file("revision.py",revision)
    second=await service.train_research_model("Reference","revision.py","","Linear relationship",
        parent_experiment_id=first["experiment"]["id"],variant="candidate")
    assert second["experiment"]["validation"]["score"]<1e-8
    assert second["experiment"]["comparison"]["improved_on_validation"]
    assert second["experiment"]["evaluation_sha256"]==first["experiment"]["evaluation_sha256"]
