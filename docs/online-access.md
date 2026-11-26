# Online conversations and administration

Agent and Researcher share online admission. The default pool is **three new conversations per calendar day**, using `Asia/Shanghai` as the day boundary. A draft, an upload or a mode switch does not consume a conversation. The first accepted submission does. Agent follow-ups and retries reuse the same admission. A new Researcher report is a new conversation; its stable request ID deduplicates retries across `/jobs`, `/generate` and `/generate/stream`. Deleting a conversation does not refund it. Admission survives server restarts and uses a SQLite write transaction, so simultaneous requests cannot exceed the allowance.

Anonymous users share the free allowance for their IP and retain a separate browser identity. Both checks must pass. IPv6 addresses are grouped by /64. This approximates a person without requiring registration; it cannot prove a real person's identity, and people behind the same institution NAT share the free pool. A person who changes both network and browser cannot be reliably identified without login/verification.

An administrator can issue one email-bound key with **additional** daily conversations, an expiry, recipient details and notes. The email/key's total is shared across devices; rotation, revocation or changing the allowance does not erase usage. Extra allowance can still be used when an institution's free IP pool is exhausted. Example: `extra=7` permits at most ten daily conversations for that key, with the anonymous IP/browser pool still applying to the three free slots. Unused allowance does not roll over. Keys are random bearer credentials: keep them private.

Settings → Extra allowance validates and saves the email and key in this browser's local storage, as requested. Only same-origin quota/chat/report requests carry these credentials; they are never included in model prompts. Clear saved key removes the browser copy. Expired/revoked keys cannot start new work. Existing admitted conversations remain accessible under their existing ownership checks.

The quota dialog includes an application form (institution, name, email, contact, requested amount and purpose). **Compose request email** opens a draft addressed to `tanyang.august@sjtu.edu.cn`; the person sends it in their email application. There is no SMTP delivery claim or background email transmission.

## Administrator setup

The direct routes `/admin`, `/en/admin`, `/zh/admin` are deliberately absent from normal navigation. The API requires an authenticated administrator regardless of how the URL was discovered.

```bash
conda activate venus
python scripts/admin_access.py bootstrap --email tanyang.august@sjtu.edu.cn
```

The command prompts for a password and never resets an existing administrator. No default password is embedded in code. Passwords use salted scrypt; random quota keys and opaque administrator session tokens are stored as hashes. Administrator cookies are HttpOnly, SameSite=Strict, and Secure under HTTPS; sessions expire after eight hours. Password changes and account disable/delete revoke existing sessions. Login attempts are throttled by IP and account. Mutating management endpoints require same-origin requests plus a non-simple application header. Administrators can create/edit/disable/delete other administrators; deleting or disabling the signed-in administrator or the last active administrator is rejected. Key deletion is revocation: a disabled record is retained to preserve usage and operational history. Reissuing a key invalidates the previous key immediately.

The database defaults to `~/.local/share/venusfactory/access/<checkout-hash>/access.sqlite3` (directory mode 0700, file mode 0600). Back it up with the service state; **do not delete it on redeploy**, because it contains users and current/historical admission records. Set `VENUS_ACCESS_DB` to a stable absolute path when deploying another checkout; do not copy credentials via tracked files. If running multiple API processes, point them at this same database on local storage. SQLite is not a multi-host distributed quota backend.

## Reverse proxy and configuration

Run Uvicorn with `--no-proxy-headers` and configure `VENUS_TRUSTED_PROXIES` with only the immediate trusted proxy addresses/CIDRs. The application walks the forwarded chain from the trusted peer and chooses the first untrusted address. Never trust arbitrary client `X-Forwarded-For`, and do not set an all-address trust list. Configure the reverse proxy to set/append XFF, serve HTTPS, and block direct public access to the backend port. For TLS termination, the immediate trusted proxy must set `X-Forwarded-Proto: https`; the application accepts it only from `VENUS_TRUSTED_PROXIES`, keeping authentication cookies Secure. Local HTTP development is supported.

Relevant environment settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEBUI_V2_MODE` | existing deployment setting | `online` enforces admission and disables Scientist execution |
| `WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT` | `3` | Shared free new-conversation pool |
| `VENUS_QUOTA_TIMEZONE` | `Asia/Shanghai` | Calendar day boundary |
| `VENUS_ACCESS_DB` | checkout-scoped path above | Persistent credentials and ledger |
| `VENUS_TRUSTED_PROXIES` | empty | Proxies allowed to supply forwarded client addresses |

## Scientist runs locally

Start or resume Scientist studies in a local deployment. Online access to existing studies and artifacts remains subject to session ownership. Online users can start Agent conversations and Researcher reports.

## Verification

`tests/web/test_online_access.py` exercises atomic concurrent admission, shared modes and aliases, retry idempotency, calendar reset, IP/browser identity, email key sharing, expiration/rotation/revocation, administrator authentication/CSRF/throttling and online admission enforcement. It does not call external models. Existing report, Scientist supervisor and Agent transport regressions must also pass before publishing a preview.
