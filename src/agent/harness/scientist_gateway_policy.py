"""Request controls for live Scientist roles; historical protocols are unchanged."""
import copy
import hashlib
import json
from contextlib import contextmanager

from agent.rsi.engine.model_gateway import GatewayPolicy
from .responses_gateway import ResponsesPolicy


class RoleOutputPolicy:
    @contextmanager
    def role_output(self, specification):
        if getattr(self, "_role_output", None) is not None:
            raise ValueError("A scientific role output is already active")
        self._role_output = copy.deepcopy(specification)
        try:
            yield
        finally:
            self._role_output = None

    def apply(self, payload):
        spec = getattr(self, "_role_output", None)
        if spec is not None:
            if (payload.get("tools") != [spec["tool"]]
                    or payload.get("tool_choice") != spec["choice"]
                    or ("parallel_tool_calls" in payload and payload["parallel_tool_calls"] is not False)
                    or payload.get("stream") is not False
                    or any(k in payload for k in ("functions", "function_call"))):
                raise ValueError("Scientific role output cannot expose executable tools")
            clean = {k: v for k, v in payload.items()
                     if k not in {"tools", "tool_choice", "parallel_tool_calls"}}
            sent, metadata = super().apply(clean)
            sent.update(tools=[copy.deepcopy(spec["tool"])],
                        tool_choice=copy.deepcopy(spec["choice"]))
            if "parallel_tool_calls" in payload:
                sent["parallel_tool_calls"] = False
            else:
                sent.pop("parallel_tool_calls", None)
            metadata.update(
                tools_before=[spec["name"]], tools_sent=[spec["name"]],
                tool_schema_sha256=hashlib.sha256(json.dumps(sent["tools"], sort_keys=True).encode()).hexdigest(),
                role_output_policy="typed_scientist_output.v1",
                role_output=spec["role"], parallel_tool_calls_sent=sent.get("parallel_tool_calls"))
        else:
            output_names = {"return_research_direction", "return_research_plan", "return_critic_review"}
            for tool in payload.get("tools") or []:
                if isinstance(tool, dict):
                    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
                    if function.get("name") in output_names:
                        raise ValueError("Scientific role output is unavailable outside its phase")
            sent, metadata = super().apply(payload)
            if payload.get("tool_choice") == "none" and not sent.get("tools"):
                sent["tool_choice"] = "none"
                metadata["role_tool_policy"] = "explicit_no_tools.v1"
        encoded = json.dumps(sent, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode()
        metadata.update(request_sha256=hashlib.sha256(encoded).hexdigest(),
                        request_bytes=len(encoded))
        return sent, metadata


class ScientistGatewayPolicy(RoleOutputPolicy, GatewayPolicy):
    pass


class ScientistResponsesPolicy(RoleOutputPolicy, ResponsesPolicy):
    pass
