#!/usr/bin/env python3
"""Audit a resolved Python profile against OSV; report all known and new findings.

The input must contain exact pins produced by uv pip compile for one platform.
No application code, credentials, model weights or installed environment is read.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/security/open-model-advisories.json"
API = "https://api.osv.dev/v1/querybatch"
VERSION = r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?"
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(" + VERSION + r")")
ADVISORY = re.compile(r"(?:GHSA-[a-z0-9-]+|PYSEC-[0-9-]+|CVE-[0-9-]+)")


def read_pins(path: Path) -> list[tuple[str, str]]:
    pins: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN.fullmatch(line)
        if not match:
            raise ValueError(f"{path.name}:{number}: expected one exact resolved pin")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in pins and pins[name] != version:
            raise ValueError(f"{path.name}:{number}: conflicting versions for {name}")
        pins[name] = version
    if not pins:
        raise ValueError("The resolved dependency profile is empty")
    return sorted(pins.items())


def post(payload: dict) -> dict:
    request = Request(API, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("OSV response exceeds the audit limit")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("OSV returned an invalid response")
            return value
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("OSV audit could not complete")


def query(pins: list[tuple[str, str]], send=post) -> dict[tuple[str, str], set[str]]:
    findings = {pin: set() for pin in pins}
    for offset in range(0, len(pins), 100):
        pending = [(pin, "") for pin in pins[offset:offset + 100]]
        seen = set()
        for _ in range(50):
            queries = []
            for (name, version), token in pending:
                item = {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
                if token:
                    item["page_token"] = token
                queries.append(item)
            reply = send({"queries": queries})
            results = reply.get("results")
            if not isinstance(results, list) or len(results) != len(pending):
                raise ValueError("OSV response does not cover every requested package")
            following = []
            for (pin, _), row in zip(pending, results):
                if not isinstance(row, dict) or row.get("error"):
                    raise ValueError("OSV returned an incomplete package result")
                vulnerabilities = row.get("vulns", [])
                if not isinstance(vulnerabilities, list):
                    raise ValueError("OSV returned invalid vulnerability records")
                for vulnerability in vulnerabilities:
                    identifier = vulnerability.get("id") if isinstance(vulnerability, dict) else None
                    if not isinstance(identifier, str) or not ADVISORY.fullmatch(identifier):
                        raise ValueError("OSV returned an invalid advisory identifier")
                    findings[pin].add(identifier)
                token = row.get("next_page_token")
                if token:
                    if not isinstance(token, str) or (pin, token) in seen:
                        raise ValueError("OSV pagination did not advance")
                    seen.add((pin, token))
                    following.append((pin, token))
            if not following:
                break
            pending = following
        else:
            raise ValueError("OSV pagination exceeded the audit limit")
    return findings


def read_baseline(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("Invalid open-advisory inventory")
    seen = set()
    for entry in data["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid open-advisory entry")
        name, version = entry.get("name", ""), entry.get("version", "")
        if not PIN.fullmatch(f"{name}=={version}") or name != re.sub(r"[-_.]+", "-", name).lower():
            raise ValueError("Open advisories require a normalized name and exact version")
        ids = entry.get("ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and ADVISORY.fullmatch(i) for i in ids):
            raise ValueError("Open advisories require explicit advisory identifiers")
        if not entry.get("reason") or not entry.get("evidence"):
            raise ValueError("Open advisories require a reason and evidence")
        reviewed = date.fromisoformat(entry["reviewed"])
        expires = date.fromisoformat(entry["review_by"])
        if reviewed > date.today() or not 0 < (expires - reviewed).days <= 31:
            raise ValueError("Open-advisory review dates must be within 31 days")
        profiles = entry.get("profiles")
        if not isinstance(profiles, list) or not profiles or not set(profiles) <= {"linux", "macos"}:
            raise ValueError("Open advisories require an explicit supported profile")
        for profile in profiles:
            for identifier in ids:
                key = (profile, name, version, identifier)
                if key in seen:
                    raise ValueError("Duplicate open-advisory inventory entry")
                seen.add(key)
    return data["entries"]


def classify(findings: dict, baseline: list[dict], profile: str, today: date) -> dict:
    known = {}
    for entry in baseline:
        if profile in entry["profiles"]:
            for identifier in entry["ids"]:
                known[(entry["name"], entry["version"], identifier)] = entry
    rows = []
    for (name, version), identifiers in sorted(findings.items()):
        for identifier in sorted(identifiers):
            entry = known.get((name, version, identifier))
            state = "new"
            if entry:
                state = "review_overdue" if today >= date.fromisoformat(entry["review_by"]) else "known_open"
            rows.append({"name": name, "version": version, "id": identifier, "state": state})
    new = sum(row["state"] == "new" for row in rows)
    overdue = sum(row["state"] == "review_overdue" for row in rows)
    return {
        "profile": profile, "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": ("Resolved optional sandbox-client dependencies" if profile in {"cube", "e2b"} else "Resolved platform dependencies") + "; excludes independent third-party model environments",
        "package_count": len(findings), "advisory_record_count": len(rows),
        "new_count": new, "overdue_count": overdue,
        "status": "action_required" if new or overdue else "known_open" if rows else "clear",
        "findings": rows,
    }


def render(report: dict) -> str:
    lines = [f"### Python dependency audit: {report['profile']}", "",
             f"Status: **{report['status']}**. {report['package_count']} resolved packages; "
             f"{report['advisory_record_count']} advisory records.",
             "GHSA and PYSEC records may describe the same issue; this is not the GitHub alert count.", "",
             "| Package | Version | Advisory | Status |", "| --- | --- | --- | --- |"]
    for row in report["findings"]:
        lines.append(f"| {row['name']} | {row['version']} | "
                     f"[{row['id']}](https://osv.dev/vulnerability/{row['id']}) | {row['state']} |")
    lines += ["", report["scope"] + ".", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resolved", type=Path)
    parser.add_argument("--profile", choices=("linux", "macos", "cube", "e2b"), required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Fail on all open advisories, including recorded legacy model risks")
    args = parser.parse_args(argv)
    try:
        pins = read_pins(args.resolved)
        baseline = read_baseline(args.baseline)
        report = classify(query(pins), baseline, args.profile, date.today())
        code = int(report["status"] == "action_required" or (args.strict and bool(report["findings"])))
        summary = render(report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"profile": args.profile, "status": "incomplete", "error": str(exc)}
        summary = f"### Python dependency audit: {args.profile}\n\nAudit incomplete: {exc}\n"
        code = 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    if os.environ.get("GITHUB_ACTIONS") == "true" and report["status"] == "known_open":
        print("::warning title=Open model-runtime advisories::Known Torch/Transformers risks remain open. See the audit summary and docs/security/dependency-review-2026-11-08.md.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
