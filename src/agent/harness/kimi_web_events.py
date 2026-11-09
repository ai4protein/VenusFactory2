"""Translate Kimi 2.5 transcript changes into the shared Kimi event contract.

The transcript is a materialized view: final frames and reconnect snapshots
repeat content already streamed. Only previously unseen text and tool events
are emitted. Sequence gaps must be repaired by the transport before ingestion.
"""
from __future__ import annotations

import json

from agent.kimi_client import KimiEvent


class TranscriptGap(ValueError):
    pass


class KimiTranscript:
    def __init__(self):
        self.seq = 0
        self.frames = {}
        self.started_tools = set()
        self.finished_tools = set()
        self.started_turns = set()
        self.finished_turns = set()
        self.interactions = {}
        self.ended_interactions = set()

    def interaction(self, item):
        """Accept the unified wire representation or its transcript projection."""
        identity = item.get("interaction_id") or item.get("interactionId")
        if not identity:
            raise ValueError("Kimi interaction has no identity")
        request = item.get("request") or {}
        state = item.get("status") or item.get("state")
        if state != "pending":
            self.ended_interactions.add(identity)
            self.interactions.pop(identity, None)
            return
        if identity in self.ended_interactions:
            return
        self.interactions[identity] = {
            **request, "id": identity,
            "kind": item.get("kind") or item.get("interactionKind"),
            "tool_name": request.get("tool_name") or request.get("toolName"),
            "tool_input_display": request.get("tool_input_display") or request.get("display"),
        }

    def _text(self, key, frame, previous):
        text = frame.get("text", "")
        old = previous.get("text", "")
        if not isinstance(text, str) or not text.startswith(old):
            raise ValueError("Kimi transcript rewrote already emitted text")
        delta = text[len(old):]
        if not delta or (frame["kind"] == "text" and frame.get("role") != "assistant"):
            return []
        return [KimiEvent(kind="thinking" if frame["kind"] == "thinking" else "text",
                          turn_id=key[0], text=delta, raw_type="transcript.ops")]

    def _frame(self, turn, step, incoming):
        identity = incoming.get("frameId")
        if not identity:
            raise ValueError("Kimi transcript frame has no identity")
        key = (turn, step, identity)
        previous = self.frames.get(key, {})
        frame = {**previous, **incoming}
        if previous and previous.get("kind") != frame.get("kind"):
            raise ValueError("Kimi transcript frame kind changed")
        events = []
        if frame.get("kind") in {"text", "thinking"}:
            events = self._text(key, frame, previous)
        elif frame.get("kind") == "tool":
            call = frame.get("toolCallId")
            name = frame.get("name")
            if not call or not name:
                raise ValueError("Kimi transcript tool has no identity")
            if previous and (previous.get("toolCallId"), previous.get("name")) != (call, name):
                raise ValueError("Kimi transcript tool identity changed")
            args = frame.get("input")
            if args is None:
                try:
                    args = json.loads(frame.get("inputText", ""))
                except (ValueError, TypeError):
                    args = None
            terminal = frame.get("state") in {"done", "error"}
            if key not in self.started_tools and (isinstance(args, dict) or terminal):
                self.started_tools.add(key)
                events.append(KimiEvent(kind="tool_call_start", turn_id=turn,
                    tool_call_id=call, tool_name=name, tool_args=args if isinstance(args, dict) else {},
                    raw_type="transcript.ops"))
            if terminal and key not in self.finished_tools:
                self.finished_tools.add(key)
                events.append(KimiEvent(kind="tool_result", turn_id=turn, tool_call_id=call,
                    tool_name=name, tool_output=frame["output"] if frame.get("output") is not None else frame.get("error"),
                    is_error=frame["state"] == "error", raw_type="transcript.ops"))
        self.frames[key] = frame
        return events

    def _turn(self, turn):
        identity = turn.get("turnId")
        if not identity:
            raise ValueError("Kimi transcript turn has no identity")
        events = []
        if turn.get("state") != "queued" and identity not in self.started_turns:
            self.started_turns.add(identity)
            events.append(KimiEvent(kind="turn_started", turn_id=identity, raw_type="transcript.ops"))
        # Snapshot frames must be emitted before the corresponding terminal turn.
        for step in turn.get("steps", []):
            for frame in step.get("frames", []):
                events.extend(self._frame(identity, step["stepId"], frame))
        if turn.get("state") in {"completed", "failed", "cancelled"} and identity not in self.finished_turns:
            self.finished_turns.add(identity)
            events.append(KimiEvent(kind="turn_ended", turn_id=identity,
                text=turn["state"], is_error=turn["state"] != "completed", raw_type="transcript.ops"))
        return events

    def reset(self, snapshot, seq):
        if type(seq) is not int or seq < self.seq:
            raise ValueError("Kimi transcript reset moved its cursor backwards")
        events = []
        self.interactions.clear()
        for item in snapshot.get("interactions", []):
            self.interaction(item)
        for item in snapshot.get("items", []):
            if item.get("kind") == "turn":
                events.extend(self._turn(item))
        self.seq = seq
        return events

    def consume(self, batch):
        seq = batch.get("seq")
        if type(seq) is not int or seq < 0:
            raise ValueError("Kimi transcript has an invalid cursor")
        if seq <= self.seq:
            return []
        if seq != self.seq + 1:
            raise TranscriptGap("Kimi transcript has missing events")
        events = []
        for op in batch.get("ops", []):
            kind = op.get("op")
            if kind == "reset":
                events.extend(self.reset(op["snapshot"], seq))
            elif kind == "turn.upsert":
                events.extend(self._turn(op["turn"]))
            elif kind == "frame.upsert":
                events.extend(self._frame(op["turnId"], op["stepId"], op["frame"]))
            elif kind == "append":
                target = op.get("target", {})
                if target.get("type") != "frame":
                    continue
                key = (target["turnId"], target["stepId"], target["frameId"])
                frame = self.frames.get(key)
                if frame is None:
                    raise ValueError("Kimi append has no frame")
                field = "inputText" if frame["kind"] == "tool" else "text"
                before = frame.get(field, "")
                # The server is JavaScript: offsets count UTF-16 code units.
                if type(op.get("offset")) is not int or op["offset"] != len(before.encode("utf-16-le")) // 2:
                    raise ValueError("Kimi append offset differs from received text")
                events.extend(self._frame(key[0], key[1], {**frame, field: before + op["text"]}))
            elif kind == "interaction.upsert":
                self.interaction(op["interaction"])
            elif kind == "step.upsert":
                step = op["step"]
                if step.get("usage"):
                    events.append(KimiEvent(kind="other", turn_id=op["turnId"],
                        raw_type="transcript.usage", raw={"usage": step["usage"]}))
            # Prompt and internal marker updates are not assistant responses.
        self.seq = seq
        return events
