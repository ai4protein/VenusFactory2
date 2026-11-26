import pytest
from backend.web_v2.research_longform import chapter_database_sources
from backend.web_v2.research_report import knowledge_plan
from mcp_tools.tools.research import knowledge_queries as queries


@pytest.mark.asyncio
async def test_expression_preserves_units_provenance_and_server_pagination(monkeypatch):
    calls = []
    async def fetch(url, params=None, body=None):
        calls.append(body)
        return {"data": {"meta": {"dataVersion": {"year": "26", "month": "09"}},
                "target": {"id": "ENSG00000169083", "baselineExpression": {
                    "count": 7, "rows": [{"unit": "TPM", "median": 2.5,
                    "datasourceId": "GTEx", "tissueBiosampleFromSource": "Liver"}]}}}}
    monkeypatch.setattr(queries, "fetch", fetch)
    result = await queries.opentargets_query(operation="target", identifier="ENSG00000169083",
                                             resource="expression", limit=2, page=1)
    assert result["status"] == "completed" and result["has_more"] and result["total"] == 7
    assert result["records"][0]["unit"] == "TPM"
    assert result["version"]["dataVersion"]["month"] == "09"
    assert calls[0]["variables"]["page"] == {"index": 1, "size": 2}
    assert "baselineExpression(page:$page)" in calls[0]["query"]


@pytest.mark.asyncio
async def test_constraint_is_gene_level_and_unpaginated(monkeypatch):
    calls = []
    async def fetch(url, params=None, body=None):
        calls.append(body)
        return {"data": {"target": {"id": "ENSG00000169083",
                "geneticConstraint": [{"constraintType": "lof", "obs": 1, "exp": 2, "oe": 0.5}]}}}
    monkeypatch.setattr(queries, "fetch", fetch)
    result = await queries.opentargets_query(operation="target", identifier="ENSG00000169083",
                                             resource="genetic_constraint")
    assert result["records"][0]["geneticConstraint"][0]["oe"] == 0.5
    assert "page" not in calls[0]["variables"]
    with pytest.raises(ValueError):
        queries.OpenTargetsInput(operation="target", identifier="ENSG00000169083",
                                  resource="genetic_constraint", page=1)


def test_chapter_context_focuses_target_resources_including_legacy_sources():
    common = [{"id": "S1", "kind": "database", "url": "https://www.uniprot.org/uniprotkb/P00918"}]
    targets = [{"id": resource, "kind": "database",
                "url": "https://platform.opentargets.org/target/ENSG00000169083#" + resource}
               for resource in ["diseases", "drugs", "expression", "genetic_constraint", "tractability"]]
    values = common + targets
    assert [s["id"] for s in chapter_database_sources(values, "variants_modifications")] == ["S1", "genetic_constraint", "diseases"]
    assert [s["id"] for s in chapter_database_sources(values, "biophysics_production")] == ["S1", "expression"]
    assert [s["id"] for s in chapter_database_sources(values, "engineering_design")] == ["S1", "tractability", "drugs"]


def test_collection_topics_request_matching_target_resources():
    entry = {"organism": {"taxonId": 9606, "scientificName": "Homo sapiens"},
             "uniProtKBCrossReferences": [{"database": "Ensembl", "properties": [
                 {"key": "GeneId", "value": "ENSG00000169083"}]}]}
    planned = knowledge_plan(entry, ["function", "variants", "engineering"])
    assert {args.get("resource") for _, name, args in planned if name == "query_opentargets"} == {
        "diseases", "drugs", "expression", "genetic_constraint", "tractability"}
