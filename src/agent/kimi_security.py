"""Security policy for kimi tool-call approvals.

Kimi raises an approval queue entry for every tool invocation outside its
internal trust list. Our `_stream_kimi._auto_approve_all` previously approved
all of them blindly — which is fine for a single-user local install but
catastrophic for any multi-tenant deployment: kimi could happily `cat
~/.kimi-code/config.toml`, exfiltrate API keys via `curl`, `rm -rf $HOME`,
etc.

This module exposes `decide(approval, session_dir, mode)` which returns
`SecurityDecision(allowed, reason, redacted_tool_input)`. Callers (the
auto-approver and the audit logger) consume that uniformly.

Two security tiers:
  - `mode == "local"`   → loose: only block obvious foot-guns (`rm -rf ~`,
                          curl|sh, reading well-known credential paths).
  - `mode == "online"`  → strict: writes must stay inside `session_dir`;
                          Bash path arguments must too; FetchURL hosts must
                          appear in `ALLOWED_HOSTS`; secret-path reads are
                          rejected even via shell substitution.

Defense in depth — kimi-side `[permission]` rules in `~/.kimi-code/config.toml`
still apply on top of these decisions. We treat them as belt + suspenders.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from logger import get_logger

_logger = get_logger("agent.kimi_security")


# ── Well-known sensitive paths kimi must never touch ──────────────────────
# Substrings checked against absolute / normalized paths (case-insensitive).
_SECRET_PATH_SUBSTRINGS: tuple[str, ...] = (
    # Credential-bearing dotfiles. Always denied — there's no legitimate
    # reason the chat agent needs to see your DeepSeek/OpenAI/AWS keys, ssh
    # private keys, etc., even on your own machine (LLM ingesting them is
    # pure downside: the keys end up in chat history / model context).
    "/.env",
    "/.envrc",
    "/.netrc",
    "/.kimi-code/",
    "/.venusfactory/",
    "/.ssh/",
    "/.aws/",
    "/.gnupg/",
    "/.docker/config",
    "/.git-credentials",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    # NOTE: /etc/passwd, /etc/shadow, /etc/sudoers are NOT in this list.
    # /etc/passwd is world-readable (just lists usernames + home dirs);
    # /etc/shadow + /etc/sudoers require root anyway, so OS file perms
    # block any actual leak. Adding them here would just annoy local-mode
    # users who legitimately want kimi to inspect /etc/* config.
)


# ── Dangerous Bash command patterns (always denied — both modes) ──────────
# Compiled regexes; case-insensitive.
_BASH_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"rm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\s+(/|~|\$HOME|\.\.)",
        r"\bsudo\b",
        r"\bdoas\b",
        r"\bssh\s",
        r"\bscp\s",
        r"\brsync\b.*::",
        r"\bnc\s+-l",
        r"\b(curl|wget|fetch)\s[^|;]*\|\s*(bash|sh|zsh|python|perl|ruby|node)\b",
        r"\bchmod\s+(777|a\+rwx|\+s)\b",
        r"\bchown\s+",
        r":\(\)\s*\{\s*:\|:&\s*}\s*;:",
        r"\bmkfs\b",
        r"\bdd\s+if=.+of=/dev/",
        r">\s*/dev/(sd[a-z]|nvme)",
        r"\b(reboot|shutdown|halt|poweroff)\b",
        r"\biptables\s+-F",
        r"\bcrontab\s+-r",
        r"\beval\s+\$\(curl",
        r"\bbase64\s+-d\b.*\|\s*(bash|sh|zsh|python|perl|ruby|node|php)\b",

        # ── Interpreter -c/-e/-r forms: arbitrary code that bypasses our
        # argv-path scan. Force users to write a script file inside the
        # session_dir if they really need it (that path THEN gets audited).
        r"\bpython3?\b[^\n]*\s-c\s",
        r"\bperl\b[^\n]*\s-e\s",
        r"\bruby\b[^\n]*\s-e\s",
        r"\bnode\b[^\n]*\s-e\s",
        r"\bphp\b[^\n]*\s-r\s",
        r"\blua\b[^\n]*\s-e\s",
        r"\btclsh\b[^\n]*\s-c\s",
        r"\bawk\b[^\n]*\bsystem\s*\(",          # awk 'BEGIN{system(...)}'
        r"\b(bash|sh|zsh|dash|ksh)\b[^\n]*\s-c\s",   # nested shell -c

        # find -exec / -execdir lets you smuggle a shell call past argv scan
        r"\bfind\b[^\n]*\s-exec(?:dir)?\b",

        # base64/xxd/hex piped to anything that executes
        r"\b(base64|xxd|hex|uudecode)\b[^\n]*\|\s*(bash|sh|zsh|python|perl|ruby|node|php|sh\s+-c)\b",

        # eval / source on dynamic input
        r"\b(eval|source|\.\s)\s*[`$]",         # eval $(...) / eval `...` / . $(...)
    )
)

# ── Online-only deny patterns: tools that egress data over the network
# bypassing our FetchURL host allowlist. Kimi already has MCP equivalents
# (query_pubmed, download_uniprot_*, etc.) that DO go through the allowlist.
_BASH_ONLINE_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(curl|wget|fetch|aria2c|httpie|http)\b",
        r"\b(nc|ncat|netcat|socat)\b",
        r"\b(dig|nslookup|host|whois|drill|delv)\b",
        r"\b(ftp|sftp|tftp|telnet)\b",
        r"\b(python3?|perl|ruby|node)\b[^\n]*\b(urllib|requests|httpx|http\.client|net::http|net/http|fetch)\b",
        r">\s*/dev/(tcp|udp)/",                # bash builtin /dev/tcp/host/port
    )
)

# ── Read-only system prefixes that bash / Read may touch even though they
# live outside the session directory (binaries, shared libs, certs, /proc).
_READ_ONLY_SYSTEM_PREFIXES: tuple[str, ...] = (
    "/usr/", "/opt/", "/lib/", "/lib64/", "/bin/", "/sbin/",
    "/etc/ssl/", "/etc/ca-certificates/",
    "/proc/", "/sys/",
    "/dev/null", "/dev/stdout", "/dev/stderr",
)


# ── Secret-path obfuscation detection (catches "cat ~/{.env,foo}" style
# tricks where the literal "/.env" never appears as a substring). We do a
# fuzzy match: if the command body contains BOTH a home-dir marker AND a
# known secret-leaf token nearby, treat it as suspicious.
_HOME_TOKENS = re.compile(r"\b(?:~|\$HOME|\$\{HOME\}|/home/[\w-]+|/root)\b", re.I)
_SECRET_LEAF_TOKENS = re.compile(
    r"(?:\.env\b|\.envrc\b|\.netrc\b"
    r"|\.kimi-code\b|\.venusfactory\b"
    r"|\.ssh\b|\.aws\b|\.gnupg\b|\.docker\b"
    r"|id_rsa\b|id_ed25519\b|id_ecdsa\b|id_dsa\b"
    r"|/keys?\.json\b)",
    re.I,
)


# ── FetchURL host allowlist (online mode only) ────────────────────────────
# These are the same upstreams the venusfactory MCP tools hit, so kimi has
# no operational need for anything else.
_ALLOWED_HOSTS: frozenset[str] = frozenset({
    # NCBI / PubMed
    "eutils.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    # RCSB / PDB
    "files.rcsb.org", "data.rcsb.org", "search.rcsb.org",
    "rcsb.org", "www.rcsb.org",
    # UniProt
    "rest.uniprot.org", "www.uniprot.org", "uniprot.org",
    # EBI: AlphaFold, InterPro, ClustalO, ChEMBL
    "alphafold.ebi.ac.uk", "alphafold.com",
    "www.ebi.ac.uk", "ebi.ac.uk",
    # KEGG, BRENDA
    "rest.kegg.jp", "kegg.jp", "www.kegg.jp",
    "www.brenda-enzymes.org", "brenda-enzymes.org",
    # STRING
    "string-db.org", "stringdb-static.org",
    # Foldseek, MMseqs2
    "search.foldseek.com",
    "search.mmseqs.com", "mmseqs.com",
    # Literature
    "www.biorxiv.org", "biorxiv.org",
    "arxiv.org", "export.arxiv.org",
    "api.openalex.org",
    "api.semanticscholar.org",
    # Code / community lookup
    "api.github.com", "raw.githubusercontent.com",
    "huggingface.co",
    # Web search backends used by query_* tools
    "api.duckduckgo.com", "duckduckgo.com",
    "api.tavily.com",
    "api.fda.gov",
})


@dataclass
class SecurityDecision:
    allowed: bool
    reason: str
    tool_name: str = ""
    redacted_input: dict[str, Any] | None = None  # safe to log


def _norm_path(p: str) -> str:
    """Resolve `~` and `..` for path comparison. Falls back to literal."""
    try:
        return str(Path(os.path.expandvars(os.path.expanduser(p))).resolve())
    except (OSError, ValueError, RuntimeError):
        return p


def _is_secret_path(path: str) -> bool:
    if not path:
        return False
    p = _norm_path(path).lower()
    return any(s in p for s in _SECRET_PATH_SUBSTRINGS)


def _is_under(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        return Path(_norm_path(path)).is_relative_to(Path(_norm_path(root)))
    except (ValueError, OSError, AttributeError):
        # is_relative_to added in Py 3.9; manual fallback just in case.
        np, nr = _norm_path(path), _norm_path(root)
        return np == nr or np.startswith(nr.rstrip("/") + "/")


def _is_allowed_host(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    if not host:
        return False
    if host in _ALLOWED_HOSTS:
        return True
    return any(host.endswith("." + h) for h in _ALLOWED_HOSTS)


def _denormalize_bash(cmd: str) -> str:
    """Strip quoting / adjacent-string concatenation so that obfuscated
    forms like `cat "~/.e""nv"` reveal their concrete `cat ~/.env` shape.

    Uses shlex to tokenize, then rejoins tokens with spaces — same letters
    a shell would see in argv[i] after quoting rules apply.
    Falls back to the literal command when shlex chokes on unbalanced quotes.
    """
    try:
        return " ".join(shlex.split(cmd, posix=True))
    except ValueError:
        return cmd


def _bash_policy(cmd: str, *, session_dir: str, mode: str) -> SecurityDecision:
    c = (cmd or "").strip()
    if not c:
        return SecurityDecision(True, "empty command", "Bash")

    # Defeat string-concat / quote-splice obfuscation by also checking the
    # denormalized (post-tokenization) form against every pattern below.
    candidates = (c, _denormalize_bash(c))

    for cand in candidates:
        for pat in _BASH_DENY_PATTERNS:
            if pat.search(cand):
                return SecurityDecision(
                    False,
                    f"bash command matches deny pattern: /{pat.pattern}/",
                    "Bash",
                )
        # Reject mentions of well-known secret paths
        cand_low = cand.lower()
        for sref in _SECRET_PATH_SUBSTRINGS:
            if sref in cand_low:
                return SecurityDecision(
                    False,
                    f"bash command references protected path: {sref}",
                    "Bash",
                )

    # Fuzzy: home-marker + secret-leaf in same command (catches brace/concat
    # obfuscation like `cat ~/{.env,foo}` or `cat "~/.e""nv"`).
    for cand in candidates:
        if _HOME_TOKENS.search(cand) and _SECRET_LEAF_TOKENS.search(cand):
            return SecurityDecision(
                False,
                "bash command combines home-dir reference with known secret leaf",
                "Bash",
            )

    if mode == "online":
        # Online: block generic network egress tools (kimi has MCP equivalents
        # that go through FetchURL host allowlist), and contain every absolute
        # / home-relative path token to session_dir.
        for cand in candidates:
            for pat in _BASH_ONLINE_DENY_PATTERNS:
                if pat.search(cand):
                    return SecurityDecision(
                        False,
                        f"online mode: bash uses network-egress tool /{pat.pattern}/ "
                        "(use mcp__venusfactory__query_* / download_* instead)",
                        "Bash",
                    )

        path_tokens = re.findall(r"(?:^|\s)(/[^\s\"'|;<>&]+|~/[^\s\"'|;<>&]*)", c)
        for tok in path_tokens:
            if _is_under(tok, session_dir):
                continue
            norm = _norm_path(tok)
            if any(norm.startswith(p.rstrip("/")) for p in _READ_ONLY_SYSTEM_PREFIXES):
                continue
            return SecurityDecision(
                False,
                f"online mode: bash references path outside session_dir: {tok}",
                "Bash",
            )

    return SecurityDecision(True, "bash policy passed", "Bash")


# ── Per-tool handlers ─────────────────────────────────────────────────────


def _decide_read_like(
    tool_name: str, args: dict[str, Any], session_dir: str, mode: str
) -> SecurityDecision:
    """Read / Glob / Grep.

    - Secret paths (.env / .ssh / id_rsa / .kimi-code / .venusfactory ...)
      are denied in BOTH modes — never makes sense for the agent to touch
      these even on the user's own machine.
    - Online mode additionally requires the path to live under session_dir
      (or a small read-only system whitelist).
    - Local mode trusts the user: read anywhere except secrets.
    """
    path = (
        args.get("file_path")
        or args.get("path")
        or args.get("pattern")
        or ""
    )
    if _is_secret_path(str(path)):
        return SecurityDecision(
            False, f"read of secret path refused: {path}", tool_name
        )
    if mode == "online" and tool_name == "Read" and path:
        norm = _norm_path(str(path))
        if not _is_under(norm, session_dir) and not any(
            norm.startswith(p.rstrip("/")) for p in _READ_ONLY_SYSTEM_PREFIXES
        ):
            return SecurityDecision(
                False,
                f"online mode: Read of {path} outside session_dir refused",
                tool_name,
            )
    return SecurityDecision(True, "read allowed", tool_name)


def _decide_write_like(
    tool_name: str, args: dict[str, Any], session_dir: str, mode: str
) -> SecurityDecision:
    """Write / Edit / NotebookEdit.

    - Secret paths denied in BOTH modes.
    - Online: must live inside session_dir (no /tmp/ escape).
    - Local: trust the user; allow writes anywhere that isn't a secret path
      (the user explicitly stated "this is my own machine" for local mode).
    """
    path = (
        args.get("file_path")
        or args.get("path")
        or args.get("notebook_path")
        or ""
    )
    if _is_secret_path(str(path)):
        return SecurityDecision(
            False, f"write to secret path refused: {path}", tool_name
        )
    if mode == "online" and not _is_under(str(path), session_dir):
        return SecurityDecision(
            False,
            f"online mode: write outside session_dir refused: {path}",
            tool_name,
        )
    return SecurityDecision(True, "write allowed", tool_name)


def _decide_fetchurl(
    tool_name: str, args: dict[str, Any], mode: str
) -> SecurityDecision:
    """FetchURL — host allowlist applies in online mode; local mode trusts
    the user and lets any URL through."""
    url = str(args.get("url") or "")
    if mode == "online" and not _is_allowed_host(url):
        return SecurityDecision(
            False,
            f"online mode: FetchURL host not in allowlist: {url}",
            tool_name,
        )
    return SecurityDecision(True, "fetchurl allowed", tool_name)


# ── Entry point ───────────────────────────────────────────────────────────

def decide(
    approval: dict[str, Any],
    *,
    session_dir: str,
    mode: str = "local",
) -> SecurityDecision:
    """Make an allow/deny decision for one kimi pending approval.

    `approval` is the dict shape returned by GET /api/v1/sessions/{id}/approvals
    — has `tool_name`, `action`, `tool_input_display`.
    `mode` is "local" (loose) or "online" (strict).
    """
    tool = str(approval.get("tool_name") or "")
    args = approval.get("tool_input_display") or {}
    if not isinstance(args, dict):
        args = {}

    # 1. venusfactory MCP tools — always allow. They go through our own
    # FastMCP server which already runs in our process, so they can't escape
    # our trust boundary further.
    if tool.startswith("mcp__venusfactory__"):
        return SecurityDecision(True, "venusfactory MCP allowlist", tool)

    # 2. Read-like kimi builtins
    if tool in ("Read", "Glob", "Grep"):
        return _decide_read_like(tool, args, session_dir, mode)

    # 3. Write-like
    if tool in ("Write", "Edit", "NotebookEdit"):
        return _decide_write_like(tool, args, session_dir, mode)

    # 4. Bash
    if tool == "Bash":
        cmd = str(args.get("command") or args.get("cmd") or "")
        return _bash_policy(cmd, session_dir=session_dir, mode=mode)

    # 5. FetchURL / WebFetch
    if tool in ("FetchURL", "WebFetch"):
        return _decide_fetchurl(tool, args, mode)

    # 6. Skill / TaskCreate / planning tools — allow (kimi-internal, no FS)
    if tool in (
        "Skill", "TaskCreate", "TaskUpdate", "TaskGet", "TaskList",
        "EnterPlanMode", "ExitPlanMode",
        "CreateGoal", "GetGoal", "SetGoalBudget", "UpdateGoal",
        "Bash",  # already handled, kept for clarity
    ):
        return SecurityDecision(True, "kimi-internal tool allowed", tool)

    # 7. Unknown tools — deny by default in online mode, allow in local.
    if mode == "online":
        return SecurityDecision(
            False, f"unknown tool not in allowlist (online mode): {tool}", tool
        )
    return SecurityDecision(True, f"unknown tool allowed (local mode): {tool}", tool)
