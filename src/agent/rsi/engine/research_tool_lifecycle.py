"""Add typed acceptance gates without changing the historical numerical ABI."""

from jsonschema import ValidationError

from .paper_step import PaperStepLifecycle
from .research_tool_contract import validate_contract, validate_result


class ResearchToolLifecycle(PaperStepLifecycle):
    def __init__(self, root, definition, contract):
        self.contract = validate_contract(contract, definition)
        super().__init__(root, definition)

    def _verify_assessment(self, identity):
        verdict = super()._verify_assessment(identity)
        for case, row in zip(self.definition["cases"]["acceptance"], verdict["cases"], strict=True):
            evidence = self._read("assessments/" + identity + "/" + case["id"] + ".json")
            try:
                validate_result(self.contract, case["inputs"], evidence["output"])
            except ValidationError:
                row["passed"] = False
        verdict["passed"] = all(row["passed"] for row in verdict["cases"])
        return verdict
