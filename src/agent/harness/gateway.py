"""Matched buffered transport: validate complete upstream evidence before delivery.

Both runtime adapters use this same transport. This intentionally differs from
older streaming pilots; do not pool timing results across transport protocols.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from aiohttp import web
from .sse_terminal import StreamTerminal
from .provider_config import request_headers
from .provider_response_diagnostics import response_evidence

from agent.rsi.engine.live_model_gateway import ModelGateway
from .live_text import PublicTextStream


def inspect_response(raw, streaming):
    if not streaming:
        obj = json.loads(raw)
        if obj.get("error") or not obj.get("choices") or not all(c.get("finish_reason") and c.get("finish_reason") != "error" for c in obj["choices"]):
            raise ValueError("Incomplete provider response")
        return obj.get("usage")
    done, finished, usage = False, set(), None
    for line in raw.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
        elif data:
            obj = json.loads(data)
            if obj.get("error"):
                raise ValueError("Provider stream error")
            usage = obj.get("usage") or usage
            for choice in obj.get("choices", []):
                if choice.get("finish_reason") == "error":
                    raise ValueError("Provider stream terminated with an error")
                if choice.get("finish_reason"):
                    finished.add(choice.get("index", 0))
    if not done or not finished:
        raise ValueError("Incomplete provider stream")
    return usage


class EvidenceGateway(ModelGateway):
    supports_free_response_evidence = True

    def __init__(self, *, request_recovery="none", on_text=None, tool_budget=None, stop_on_rate_limit=False, **kwargs):
        from .empty_response_recovery import POLICY as EMPTY_POLICY
        from .request_recovery import POLICY

        if request_recovery not in {"none", POLICY, EMPTY_POLICY}:
            raise ValueError("Unknown request recovery policy")
        from .chat_tool_budget import ChatToolBudget
        self.chat_tool_budget = ChatToolBudget(tool_budget) if tool_budget is not None else None
        self.on_text = on_text
        self.stop_on_rate_limit = stop_on_rate_limit
        self._rate_limited = False
        self.request_recovery = request_recovery
        super().__init__(**kwargs)

    async def handle(self, request):
        from .contracts import digest
        from .empty_response_recovery import POLICY as EMPTY_POLICY
        from .empty_response_recovery import eligible, response_shape
        from .request_recovery import POLICY, matches_rejection
        from .research_handoff import admit_model_request, closed_phase_response, request_scope

        try:
            scope = request_scope(self, request.headers.get("Authorization"))
        except PermissionError:
            return web.json_response({"error": "invalid relay credential"}, status=401)
        rejected = closed_phase_response(self, scope)
        if rejected is not None:
            return rejected
        if request.path != "/v1/chat/completions" or request.method != "POST":
            return web.json_response({"error": "unsupported model route"}, status=404)
        from .research_handoff import closed_response
        handoff = closed_response(self.policy, self.record)
        if handoff is not None:
            return handoff
        try:
            incoming = await request.json()
            rejected = closed_phase_response(self, scope)
            if rejected is not None:
                return rejected
            if not isinstance(incoming, dict) or type(incoming.get("stream", False)) is not bool:
                raise ValueError("Invalid streaming request")
            client_stream = incoming.get("stream", False)
            # DeepSeek Harness attaches local plugin inventory to its OpenAI
            # request. It is not an inference parameter and must not leave the host.
            local_fields = [key for key in ('dsh_plugin_packages',) if key in incoming]
            for key in local_fields:
                incoming.pop(key)
            if self.on_text is not None:
                incoming = {**incoming, "stream": True}
            budget_info = {}
            if self.chat_tool_budget is not None:
                incoming, budget_info = self.chat_tool_budget.prepare(
                    incoming, final_call=self.calls >= self.policy.max_calls - 1)
            payload, metadata = self.policy.apply(incoming)
            metadata.update(budget_info)
            if local_fields:
                metadata['local_only_fields'] = local_fields
            from .scientist_policy import request_policy_witness

            metadata.update(request_policy_witness(payload))
        except (ValueError, KeyError, TypeError):
            self.record({"event": "policy_rejected"})
            return web.json_response({"error": "request policy rejected"}, status=400)
        if self._rate_limited:
            self.record({"event": "upstream_retry_suppressed", "status": 429})
            return web.json_response({"error": "upstream_http_429"}, status=400)
        from urllib.parse import urlsplit
        from backend.web_v2.job_queue import resource_slot
        provider=(urlsplit(self.upstream_url).hostname or 'provider').lower()
        async with resource_slot('api:'+provider,kind='model-request'):
            deadline = time.monotonic() + self.timeout_s
            logical_call, previous = None, None
            for attempt in range(1, 3 if self.request_recovery != "none" else 2):
                rejected = closed_phase_response(self, scope)
                if rejected is not None:
                    return rejected
                handoff = closed_response(self.policy, self.record)
                if handoff is not None:
                    return handoff
                if time.monotonic() >= deadline:
                    self.record({"event": "request_deadline_exhausted", "logical_call": logical_call})
                    return web.json_response({"error": "logical request deadline exhausted"}, status=502)
                rejected = admit_model_request(self, scope)
                if rejected is not None:
                    return rejected
                call, started = self.calls, time.monotonic()
                logical_call = logical_call or call
                recovery = (
                    {
                        "recovery_policy": self.request_recovery,
                        "logical_call": logical_call,
                        "attempt": attempt,
                        "retry_of": previous,
                    }
                    if self.request_recovery != "none"
                    else {}
                )
                self.record(
                    {
                        "event": "request",
                        "call": call,
                        "total_deadline_s": self.timeout_s,
                        "transport": "web_live_text_validated_tools_v1" if self.on_text is not None else "buffered_complete_response_v1",
                        **metadata,
                        **recovery,
                    }
                )
                status, usage, completed, error_type = None, None, False, None
                retry = False
                shape = None
                provider_evidence = None
                raw = bytearray()
                validation_stage = "transport"
                try:
                    async with asyncio.timeout(max(0, deadline - time.monotonic())):
                        async with self.client.stream(
                            "POST",
                            self.upstream_url + "/chat/completions",
                            json=payload,
                            headers=request_headers(self.upstream_url, self._api_key),
                        ) as response:
                            status = response.status_code
                            if status != 200:
                                evidence = await self.record_upstream_rejection(response, call)
                                retry = self.request_recovery == POLICY and attempt == 1 and matches_rejection(evidence)
                                if not retry:
                                    # Stop native SDK quota retries in interactive and
                                    # autonomous research runs; retain the real upstream
                                    # status in evidence for an explicit later resume.
                                    client_status = 400 if status == 429 and (self.on_text is not None or self.stop_on_rate_limit) else status
                                    self._rate_limited = client_status != status
                                    return web.json_response({"error": "upstream_http_" + str(status)}, status=client_status)
                            else:
                                raw = bytearray()
                                preview = PublicTextStream(self.on_text, call) if self.on_text is not None else None
                                terminal = StreamTerminal('chat') if payload.get('stream') else None
                                async for chunk in response.aiter_bytes():
                                    raw.extend(chunk)
                                    if len(raw) > 16_000_000:
                                        raise ValueError("Provider response exceeds byte budget")
                                    if preview is not None:
                                        await preview.feed(chunk)
                                    if terminal is not None and terminal.feed(chunk):
                                        break
                                validation_stage = "response_completion"
                                usage = inspect_response(bytes(raw), bool(payload.get("stream")))
                                validation_stage = "free_pricing"
                                provider_evidence = self.verify_provider_response(bytes(raw), bool(payload.get("stream")))
                                completed = True
                                if self.request_recovery == EMPTY_POLICY:
                                    shape = response_shape(bytes(raw), bool(payload.get("stream")))
                                    self.record({"event": "response_shape", "call": call, **shape})
                                    retry = attempt == 1 and eligible(shape)
                                if not retry:
                                    from .tool_response_names import normalize
                                    delivered = preview.completion() if preview is not None and not client_stream else bytes(raw)
                                    validation_stage = "tool_interface"
                                    delivered, aliases = normalize(delivered, client_stream, payload.get('tools', []))
                                    if aliases:
                                        self.record({'event':'tool_name_normalized','call':call,'aliases':aliases})
                                    if self.chat_tool_budget is not None:
                                        validation_stage = "tool_budget"
                                        delivered, budget_info = self.chat_tool_budget.deliver(delivered, client_stream)
                                        self.record({'event': 'chat_tool_budget', 'call': call, **budget_info})
                                    return web.Response(
                                        body=delivered,
                                        content_type="text/event-stream" if client_stream else "application/json",
                                        status=200,
                                    )
                except asyncio.CancelledError:
                    error_type = "CancelledError"
                    self.record({"event": "transport_error", "call": call, "error_type": error_type})
                    raise
                except (httpx.HTTPError, ValueError, OSError, TimeoutError) as exc:
                    completed = False
                    error_type = type(exc).__name__
                    self.record({"event": "transport_error", "call": call, "error_type": error_type,
                                 "validation_stage": validation_stage,
                                 **response_evidence(bytes(raw), bool(payload.get("stream")))})
                    return web.json_response({"error": "incomplete upstream response"}, status=502)
                finally:
                    if not usage and raw:
                        from .provider_usage import reported_usage
                        usage = reported_usage(bytes(raw), bool(payload.get("stream")))
                    self.record(
                        {
                            "event": "response",
                            "call": call,
                            "status": status,
                            "completed": completed,
                            "error_type": error_type,
                            "usage": usage,
                            **({"response_completion": response_evidence(bytes(raw), bool(payload.get("stream")))} if raw else {}),
                            **({"provider_evidence": provider_evidence} if provider_evidence is not None else {}),
                            **(
                                {"response_sha256": shape["response_sha256"], "response_shape_sha256": digest(shape)}
                                if shape is not None
                                else {}
                            ),
                            "total_deadline_s": self.timeout_s,
                            "wall_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                if retry:
                    previous = call
                    self.record(
                        {
                            "event": "retry_scheduled",
                            "failed_call": call,
                            "logical_call": logical_call,
                            "delay_seconds": 0.5,
                        }
                    )
                    # Backoff consumes the original logical request deadline. The
                    # retry is another charged physical attempt, with identical payload.
                    await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))
            raise AssertionError("Bounded request loop must return a response")
