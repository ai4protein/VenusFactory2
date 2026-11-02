"""Live research transport with current authorization.

The historical transport module remains byte-identical for pinned evidence
replay. All current execution entry points use this wrapper.
"""

from __future__ import annotations

import httpx

from .model_access import ModelAccessDenied, authorize_model
from .model_gateway import ModelGateway as HistoricalModelGateway


class _AuthorizedPolicy:
    def __init__(self, policy, authorize, upstream_url):
        self.policy = policy
        self.authorize = authorize
        self.upstream_url = upstream_url

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def apply(self, payload):
        payload, metadata = self.policy.apply(payload)
        metadata["model_access"] = self.authorize()
        from .openrouter_policy import enforce_free_request, is_openrouter

        if is_openrouter(self.upstream_url):
            if hasattr(self.policy, "max_output_tokens"):
                from .openrouter_policy import enforce_free_responses_request

                payload, metadata = enforce_free_responses_request(payload, metadata)
            else:
                payload, metadata = enforce_free_request(payload, metadata)
        from agent.harness.provider_config import adapt_request
        protocol = "responses" if hasattr(self.policy, "max_output_tokens") else "chat_completions"
        adapted = adapt_request(payload, self.upstream_url, protocol)
        if adapted != payload:
            import hashlib
            import json
            payload = adapted
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            metadata.update(request_sha256=hashlib.sha256(encoded).hexdigest(), request_bytes=len(encoded),
                            temperature=payload.get("temperature"), thinking_sent=payload.get("thinking"),
                            reasoning_effort_sent=payload.get("reasoning_effort"),
                            provider_parameters="configured_official_defaults.v1")
        return payload, metadata


class ModelGateway(HistoricalModelGateway):
    supports_free_response_evidence = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy = _AuthorizedPolicy(self.policy, self.authorize_model_access, self.upstream_url)
        self._provider_access_fault = False
        self._usage_observations = {}

    def record(self, value):
        super().record(value)
        if value.get('event') not in {'request', 'response'}: return
        from backend.web_v2.usage_store import Observation
        from urllib.parse import urlsplit
        call = value.get('call')
        if value['event'] == 'request':
            self._usage_observations[call] = Observation('model', self.policy.model, provider=urlsplit(self.upstream_url).hostname or '')
        elif call in self._usage_observations:
            self._usage_observations.pop(call).finish(usage=value.get('usage'), failed=not value.get('completed'), error_type=value.get('error_type') or '')

    def authorize_model_access(self):
        from .openrouter_policy import is_openrouter

        if self._provider_access_fault or (
            is_openrouter(self.upstream_url) and not self.supports_free_response_evidence
        ):
            self.record({"event": "model_access_rejected"})
            raise ModelAccessDenied("OpenRouter requires intact buffered zero-cost response evidence")
        # Only explicitly injected in-memory test transports skip live admission.
        if type(self.transport) is httpx.MockTransport:
            return {"scope": "offline_mock"}
        responses = hasattr(self.policy, "max_output_tokens")
        try:
            return authorize_model(
                model=self.policy.model,
                base_url=self.upstream_url,
                protocol="responses" if responses else "chat_completions",
                output_tokens=self.policy.max_output_tokens if responses else self.policy.max_tokens,
            )
        except ModelAccessDenied:
            self.record({"event": "model_access_rejected"})
            raise

    def verify_provider_response(self, raw, streaming):
        from .openrouter_policy import is_openrouter, verify_free_response

        if not is_openrouter(self.upstream_url):
            return None
        try:
            if hasattr(self.policy, "max_output_tokens"):
                from .openrouter_policy import verify_free_responses_response

                return verify_free_responses_response(raw, streaming, self.policy.model)
            return verify_free_response(raw, streaming, self.policy.model)
        except (ValueError, TypeError, KeyError):
            self._provider_access_fault = True
            self.record({"event": "free_pricing_rejected"})
            raise ValueError("OpenRouter zero-cost response evidence failed") from None

    async def __aenter__(self):
        self.authorize_model_access()
        return await super().__aenter__()
