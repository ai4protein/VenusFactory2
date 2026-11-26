"""Protocol doubles for report transport tests; these do not validate biology."""
import functools
import json
from backend.web_v2.research_source_review import SYSTEM


def with_source_review(writer):
    @functools.wraps(writer)
    async def complete(model, messages, tokens, **_review_metadata):
        if messages[0]["content"] != SYSTEM:
            return await writer(model, messages, tokens)
        request=json.loads(messages[1]["content"])
        rows=[]
        for unit in request["units"]:
            source=next((s for s in request["sources"] if s["id"] in unit["citations"]
                         and any(len(p["text"].strip())>=4 for p in s["passages"])),None)
            passage=next((p for p in source["passages"] if len(p["text"].strip())>=4),None) if source else None
            rows.append({"unit_id":unit["id"],"classification":"factual",
                         "verdict":"supported" if passage else "unsupported",
                         "issues":[] if passage else [{"claim":unit["text"],"reason":"No cited fixture source"}],
                         "evidence":[{"source_id":source["id"],"passage_id":passage["id"],"quote":passage["text"]}]
                         if passage else []})
        return json.dumps({"assessments":rows}),False
    return complete
