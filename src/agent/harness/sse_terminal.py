"""Recognize complete SSE terminal frames without waiting for HTTP EOF.

This is a transport boundary only: gateways still validate the full response,
model identity and tool arguments before returning anything to a harness.
"""
from __future__ import annotations
import json


class StreamTerminal:
    def __init__(self, protocol="chat"):
        self.protocol=protocol
        self.pending=b""
        self.data=[]

    def feed(self, chunk):
        self.pending+=chunk
        while b"\n" in self.pending:
            line,self.pending=self.pending.split(b"\n",1)
            line=line.rstrip(b"\r")
            if line.startswith(b"data:"):
                self.data.append(line[5:].lstrip(b" "))
            elif not line and self.data:
                data,self.data=b"\n".join(self.data),[]
                if self.protocol=="chat":
                    if data==b"[DONE]":
                        return True
                else:
                    try:
                        event=json.loads(data)
                    except ValueError:
                        continue
                    if isinstance(event,dict) and event.get("type")=="response.completed":
                        return True
        return False
