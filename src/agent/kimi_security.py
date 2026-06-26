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
# bypassing our FetchURL host allowlist, OR enumerate environment variables
# that would leak the host fastapi's API keys (kimi inherits parent env).
# Kimi already has MCP equivalents (query_pubmed, download_uniprot_*, etc.)
# for legitimate network calls.
_BASH_ONLINE_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Network egress
        r"\b(curl|wget|fetch|aria2c|httpie|http)\b",
        r"\b(nc|ncat|netcat|socat)\b",
        r"\b(dig|nslookup|host|whois|drill|delv)\b",
        r"\b(ftp|sftp|tftp|telnet)\b",
        r"\b(python3?|perl|ruby|node)\b[^\n]*\b(urllib|requests|httpx|http\.client|net::http|net/http|fetch)\b",
        r">\s*/dev/(tcp|udp)/",                # bash builtin /dev/tcp/host/port

        # Environment-variable enumeration (would leak parent fastapi's keys)
        r"^\s*(env|printenv)\b",                                       # `env` / `printenv` standalone
        r"\b(env|printenv)\s*\|\s*",                                   # piping enum to grep/etc
        r"\bdeclare\s+-p\b",                                           # bash declare -p (dumps vars)
        r"\bset\b\s*\|\s*(grep|head|tail|sed|awk|sort|less|more|cat)", # set | grep secrets
        r"\$\{?(?:OPENAI|ANTHROPIC|DEEPSEEK|MOONSHOT|DMX|ZHIPU|DASHSCOPE|GOOGLE|GEMINI|AWS|GCP|AZURE|HF|HUGGINGFACE)_?[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)",
        r"\$\{?KIMI_[A-Z_]*",                                          # any $KIMI_* refs (BIN/PORT included — kimi shouldn't introspect its own config)
        r"\$\{?(?:CHAT_|WEBUI_V2_)[A-Z_]*(?:SECRET|TOKEN|KEY)",        # our own app secrets
    )
)

# ── Read-only system prefixes that bash / Read may touch even though they
# live outside the session directory.
# Local mode is loose (your own machine, you can read /proc); online mode
# is strict — /proc is a *huge* info-leak surface (environ, cmdline, maps,
# net/route, status of any pid you can stat). Allow only a handful of
# read-only system roots needed to invoke binaries / load shared libs.
_READ_ONLY_SYSTEM_PREFIXES_LOCAL: tuple[str, ...] = (
    "/usr/", "/opt/", "/lib/", "/lib64/", "/bin/", "/sbin/",
    "/etc/ssl/", "/etc/ca-certificates/",
    "/proc/", "/sys/",
    "/dev/null", "/dev/stdout", "/dev/stderr",
)
_READ_ONLY_SYSTEM_PREFIXES_ONLINE: tuple[str, ...] = (
    "/usr/lib/", "/usr/lib64/", "/usr/share/",
    "/lib/", "/lib64/",
    "/etc/ssl/certs/", "/etc/ca-certificates/",
    "/dev/null", "/dev/stdout", "/dev/stderr",
    # NOTE: /proc/cpuinfo and /proc/meminfo are intentionally NOT here —
    # legitimate uses are rare and any /proc whitelist is a hole. If you
    # need them, write the value into the kimi system prompt at session
    # create time (e.g., available_cpus=N), don't let kimi read /proc.
)


def _read_only_prefixes(mode: str) -> tuple[str, ...]:
    return _READ_ONLY_SYSTEM_PREFIXES_ONLINE if mode == "online" else _READ_ONLY_SYSTEM_PREFIXES_LOCAL


# ── /proc info-leak paths explicitly denied in online mode regardless of
# any whitelist. Catches `cat /proc/$$/environ` (parent fastapi env vars)
# and similar pid-introspection attacks.
_PROC_INFO_LEAK_RE = re.compile(
    r"^/proc/(?:self|thread-self|\d+)/"
    r"(?:environ|cmdline|status|stat|maps|smaps|mem|mounts|fd(?:/|$)|task(?:/|$)|"
    r"comm|sched|net/|root(?:/|$)|cwd(?:/|$)|exe(?:/|$))",
    re.IGNORECASE,
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

        prefixes = _read_only_prefixes("online")
        path_tokens = re.findall(r"(?:^|\s)(/[^\s\"'|;<>&]+|~/[^\s\"'|;<>&]*)", c)
        for tok in path_tokens:
            norm = _norm_path(tok)
            # /proc info-leak paths always denied, even if they would match
            # one of the read-only prefixes — environ/cmdline trump perms.
            if _PROC_INFO_LEAK_RE.match(norm):
                return SecurityDecision(
                    False,
                    f"online mode: bash references /proc info-leak path: {tok}",
                    "Bash",
                )
            if _is_under(tok, session_dir):
                continue
            if any(norm.startswith(p.rstrip("/")) for p in prefixes):
                continue
            return SecurityDecision(
                False,
                f"online mode: bash references path outside session_dir: {tok}",
                "Bash",
            )

    return SecurityDecision(True, "bash policy passed", "Bash")


# ── Per-tool handlers ─────────────────────────────────────────────────────


# Grep patterns that look like they're fishing for secrets — denied online.
_GREP_SECRET_PATTERN_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|token|password|bearer|"
    r"BEGIN\s+(?:OPENSSH|RSA|DSA|EC|PGP)\s+PRIVATE\s+KEY|"
    r"aws_access_key|aws_secret|gcp_credentials|private_key)"
)


def _decide_read_like(
    tool_name: str, args: dict[str, Any], session_dir: str, mode: str
) -> SecurityDecision:
    """Read / Glob / Grep.

    - Secret paths (.env / .ssh / id_rsa / .kimi-code / .venusfactory ...)
      are denied in BOTH modes — never makes sense for the agent to touch
      these even on the user's own machine.
    - Online mode additionally requires Read/Glob/Grep paths to live under
      session_dir (or a small read-only system whitelist that excludes /proc).
    - Online mode Grep patterns are also screened — `Grep("api_key", "/")`
      can side-channel-leak the existence/location of secrets even without
      reading them.
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
    # Online: containment applies to Read, Glob AND Grep — the previous
    # version only checked Read, letting `Glob("/**/*.env")` enumerate the
    # whole filesystem for sensitive files.
    if mode == "online" and tool_name in ("Read", "Glob", "Grep") and path:
        norm = _norm_path(str(path))
        if _PROC_INFO_LEAK_RE.match(norm):
            return SecurityDecision(
                False,
                f"online mode: {tool_name} of /proc info-leak path refused: {path}",
                tool_name,
            )
        if not _is_under(norm, session_dir) and not any(
            norm.startswith(p.rstrip("/")) for p in _read_only_prefixes(mode)
        ):
            return SecurityDecision(
                False,
                f"online mode: {tool_name} of {path} outside session_dir refused",
                tool_name,
            )
    # Online Grep: also screen the pattern for secret-fishing.
    if mode == "online" and tool_name == "Grep":
        pat = str(args.get("pattern") or "")
        if _GREP_SECRET_PATTERN_RE.search(pat):
            return SecurityDecision(
                False,
                f"online mode: Grep pattern looks like secret-fishing: {pat[:80]}",
                tool_name,
            )
    return SecurityDecision(True, "read allowed", tool_name)


def _decide_write_like(
    tool_name: str, args: dict[str, Any], session_dir: str, mode: str
) -> SecurityDecision:
    """Write / Edit / NotebookEdit.

    - Secret paths denied in BOTH modes.
    - Online: must live inside session_dir (no /tmp/ escape) AND the parent
      directory must `realpath()` inside session_dir too — defeats the
      symlink-escape attack where kimi `Write`s `<sdir>/sym → /etc/cron.d/X`
      first, then writes payload through the symlink.
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
    if mode == "online":
        if not _is_under(str(path), session_dir):
            return SecurityDecision(
                False,
                f"online mode: write outside session_dir refused: {path}",
                tool_name,
            )
        # Symlink escape check: even if the literal path looks contained,
        # its parent dir might be a symlink we (or an earlier turn) made
        # that points outside session_dir.
        try:
            parent = os.path.dirname(os.path.abspath(str(path)))
            real_parent = os.path.realpath(parent)
            if not _is_under(real_parent, session_dir):
                return SecurityDecision(
                    False,
                    f"online mode: write parent dir is a symlink escape: "
                    f"{parent} → {real_parent}",
                    tool_name,
                )
        except OSError:
            # If we can't stat, deny in online mode — fail closed.
            return SecurityDecision(
                False,
                f"online mode: cannot resolve write parent for {path}; refusing",
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


# ── Trusted MCP-tool whitelist (populated at server startup) ──────────────
# Filled by `init_trusted_mcp_tools()` after the local FastMCP server has
# registered all its tools — see api_server lifespan. Empty until then;
# in that warm-up window the prefix-match fallback below kicks in so we
# don't break startup. Once populated, only exact matches are allowed —
# guards against a malicious MCP server registering a name that prefix-
# matches `mcp__venusfactory__*`.
_TRUSTED_MCP_TOOLS: frozenset[str] = frozenset()
_TRUSTED_MCP_TOOLS_INITIALIZED: bool = False


def install_trusted_mcp_tools(names: list[str]) -> None:
    """Populate the trusted-MCP-tool set. Called once at startup.

    `names` should be the fully-qualified kimi-side tool names
    (`mcp__venusfactory__<tool_name>`) — NOT the bare FastMCP tool names.
    """
    global _TRUSTED_MCP_TOOLS, _TRUSTED_MCP_TOOLS_INITIALIZED
    cleaned = {n for n in names if n and isinstance(n, str)}
    _TRUSTED_MCP_TOOLS = frozenset(cleaned)
    _TRUSTED_MCP_TOOLS_INITIALIZED = True
    _logger.info(
        "kimi_security: loaded %d trusted MCP tool names",
        len(_TRUSTED_MCP_TOOLS),
    )


def _is_trusted_mcp_tool(tool: str) -> tuple[bool, str]:
    """Return (allowed, reason). Exact match if init'd, prefix fallback before."""
    if _TRUSTED_MCP_TOOLS_INITIALIZED:
        if tool in _TRUSTED_MCP_TOOLS:
            return True, "venusfactory MCP allowlist (exact)"
        return False, f"MCP tool not in introspection allowlist: {tool}"
    # Pre-init fallback: prefix match. Logged as warning so we notice if
    # this path stays hot past startup (would mean init never ran).
    if tool.startswith("mcp__venusfactory__"):
        _logger.warning(
            "kimi_security: MCP allowlist not initialized yet, "
            "falling back to prefix match for %s", tool,
        )
        return True, "venusfactory MCP allowlist (prefix, pre-init)"
    return False, f"unknown MCP tool: {tool}"


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

    # 1. MCP tools — must exactly match the introspected venusfactory tool
    # set. Pre-init fallback allows prefix `mcp__venusfactory__*` so the
    # very first requests during startup don't get rejected.
    if tool.startswith("mcp__"):
        ok, reason = _is_trusted_mcp_tool(tool)
        return SecurityDecision(ok, reason, tool)

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
        "TaskCreate", "TaskUpdate", "TaskGet", "TaskList",
        "EnterPlanMode", "ExitPlanMode",
        "CreateGoal", "GetGoal", "SetGoalBudget", "UpdateGoal",
    ):
        return SecurityDecision(True, "kimi-internal tool allowed", tool)

    # 6b. Skill — can execute arbitrary user-defined scripts. Deny in
    # online mode (a malicious skill bypasses every other policy here);
    # allow in local mode.
    if tool == "Skill":
        if mode == "online":
            return SecurityDecision(
                False, "online mode: Skill execution refused (skills can run arbitrary code)", tool
            )
        return SecurityDecision(True, "Skill allowed (local mode)", tool)

    # 7. Unknown tools — deny by default in online mode, allow in local.
    if mode == "online":
        return SecurityDecision(
            False, f"unknown tool not in allowlist (online mode): {tool}", tool
        )
    return SecurityDecision(True, f"unknown tool allowed (local mode): {tool}", tool)
