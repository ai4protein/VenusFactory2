"""Small stdio MCP proxy for the per-turn Web scientific tool server."""

import json
import sys
import urllib.request
from pathlib import Path


def main():
    config = json.loads(Path(__file__).with_name("connection.json").read_text())
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        try:
            req = urllib.request.Request(
                config["url"],
                data=json.dumps(request).encode(),
                headers={"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=config["timeout"]) as response:
                result = json.load(response)
        except Exception:
            result = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32603, "message": "Tool connection failed"},
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
