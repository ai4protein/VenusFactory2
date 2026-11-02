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

from agent.rsi.live_model_gateway import ModelGateway


def inspect_response(raw, streaming):
    if not streaming:
        obj = json.loads(raw)
        if obj.get("error") or not obj.get("choices") or not all(c.get("finish_reason") for c in obj["choices"]):
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
                if choice.get("finish_reason"):
                    finished.add(choice.get("index", 0))
    if not done or not finished:
        raise ValueError("Incomplete provider stream")
    return usage


class EvidenceGateway(ModelGateway):
    supports_free_response_evidence = True

    def __init__(self, *, request_recovery="none", **kwargs):
        from .empty_response_recovery import POLICY as EMPTY_POLICY
        from .request_recovery import POLICY

        if request_recovery not in {"none", POLICY, EMPTY_POLICY}:
            raise ValueError("Unknown request recovery policy")
        self.request_recovery = request_recovery
        super().__init__(**kwargs)

    async def handle(self, request):
        from .contracts import digest
        from .empty_response_recovery import POLICY as EMPTY_POLICY
        from .empty_response_recovery import eligible, response_shape
        from .request_recovery import POLICY, matches_rejection

        if request.headers.get("Authorization") != "Bearer " + self.token:
            return web.json_response({"error": "invalid relay credential"}, status=401)
        if request.path != "/v1/chat/completions" or request.method != "POST":
            return web.json_response({"error": "unsupported model route"}, status=404)
        try:
            payload, metadata = self.policy.apply(await request.json())
            from .scientist_policy import request_policy_witness

            metadata.update(request_policy_witness(payload))
        except (ValueError, KeyError, TypeError):
            self.record({"event": "policy_rejected"})
            return web.json_response({"error": "request policy rejected"}, status=400)
        deadline = time.monotonic() + self.timeout_s
        logical_call, previous = None, None
        for attempt in range(1, 3 if self.request_recovery != "none" else 2):
            if time.monotonic() >= deadline:
                self.record({"event": "request_deadline_exhausted", "logical_call": logical_call})
                return web.json_response({"error": "logical request deadline exhausted"}, status=502)
            if self.calls >= self.policy.max_calls:
                self.record({"event": "budget_rejected"})
                return web.json_response({"error": "model call budget exhausted"}, status=429)
            self.calls += 1
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
                    "transport": "buffered_complete_response_v1",
                    **metadata,
                    **recovery,
                }
            )
            status, usage, completed, error_type = None, None, False, None
            retry = False
            shape = None
            provider_evidence = None
            try:
                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                    async with self.client.stream(
                        "POST",
                        self.upstream_url + "/chat/completions",
                        json=payload,
                        headers={"Authorization": "Bearer " + self._api_key},
                    ) as response:
                        status = response.status_code
                        if status != 200:
                            evidence = await self.record_upstream_rejection(response, call)
                            retry = self.request_recovery == POLICY and attempt == 1 and matches_rejection(evidence)
                            if not retry:
                                return web.json_response({"error": "upstream_http_" + str(status)}, status=status)
                        else:
                            raw = bytearray()
                            async for chunk in response.aiter_bytes():
                                raw.extend(chunk)
                                if len(raw) > 16_000_000:
                                    raise ValueError("Provider response exceeds byte budget")
                            usage = inspect_response(bytes(raw), bool(payload.get("stream")))
                            provider_evidence = self.verify_provider_response(bytes(raw), bool(payload.get("stream")))
                            completed = True
                            if self.request_recovery == EMPTY_POLICY:
                                shape = response_shape(bytes(raw), bool(payload.get("stream")))
                                self.record({"event": "response_shape", "call": call, **shape})
                                retry = attempt == 1 and eligible(shape)
                            if not retry:
                                return web.Response(
                                    body=bytes(raw),
                                    content_type="text/event-stream" if payload.get("stream") else "application/json",
                                    status=200,
                                )
            except asyncio.CancelledError:
                error_type = "CancelledError"
                self.record({"event": "transport_error", "call": call, "error_type": error_type})
                raise
            except (httpx.HTTPError, ValueError, OSError, TimeoutError) as exc:
                error_type = type(exc).__name__
                self.record({"event": "transport_error", "call": call, "error_type": error_type})
                return web.json_response({"error": "incomplete upstream response"}, status=502)
            finally:
                self.record(
                    {
                        "event": "response",
                        "call": call,
                        "status": status,
                        "completed": completed,
                        "error_type": error_type,
                        "usage": usage,
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
