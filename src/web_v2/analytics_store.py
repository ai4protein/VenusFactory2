import json
import os
try:
    import sqlite3
    _SQLITE_IMPORT_ERROR = None
except Exception as exc:
    sqlite3 = None  # type: ignore[assignment]
    _SQLITE_IMPORT_ERROR = exc
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


@dataclass
class GeoInfo:
    country_code: str = "UNKNOWN"
    region: str = "UNKNOWN"
    city: str = "UNKNOWN"


class OfflineGeoIpResolver:
    def __init__(self, mmdb_path: Optional[str]) -> None:
        self._mmdb_path = str(mmdb_path or "").strip()
        self._reader: Any = None
        self._reader_kind = ""
        self._initialized = False

    def _init_reader(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self._mmdb_path:
            return
        path = Path(self._mmdb_path).expanduser()
        if not path.exists():
            return
        try:
            import geoip2.database  # type: ignore

            self._reader = geoip2.database.Reader(str(path))
            self._reader_kind = "geoip2"
            return
        except Exception:
            pass
        try:
            import maxminddb  # type: ignore

            self._reader = maxminddb.open_database(str(path))
            self._reader_kind = "maxminddb"
            return
        except Exception:
            self._reader = None
            self._reader_kind = ""

    def resolve(self, ip: str) -> GeoInfo:
        self._init_reader()
        ip_text = str(ip or "").strip()
        if not ip_text or ip_text == "unknown":
            return GeoInfo()
        if self._reader is None:
            return GeoInfo()
        try:
            if self._reader_kind == "geoip2":
                city = self._reader.city(ip_text)
                country_code = city.country.iso_code or "UNKNOWN"
                region = "UNKNOWN"
                if city.subdivisions and city.subdivisions.most_specific:
                    region = city.subdivisions.most_specific.name or "UNKNOWN"
                city_name = city.city.name or "UNKNOWN"
                return GeoInfo(country_code=country_code, region=region, city=city_name)
            if self._reader_kind == "maxminddb":
                node = self._reader.get(ip_text) or {}
                country_code = ((node.get("country") or {}).get("iso_code")) or "UNKNOWN"
                subdivisions = node.get("subdivisions") or []
                region = "UNKNOWN"
                if isinstance(subdivisions, list) and subdivisions:
                    first = subdivisions[0] or {}
                    names = first.get("names") or {}
                    region = names.get("en") or first.get("iso_code") or "UNKNOWN"
                city = ((node.get("city") or {}).get("names") or {}).get("en") or "UNKNOWN"
                return GeoInfo(country_code=country_code, region=region, city=city)
        except Exception:
            return GeoInfo()
        return GeoInfo()


class AnalyticsStore:
    def __init__(self) -> None:
        default_path = Path("src/data/analytics.db")
        db_path = os.getenv("WEBUI_V2_ANALYTICS_DB_PATH", str(default_path))
        self._db_path = Path(db_path).expanduser()
        self._lock = Lock()
        self._initialized = False
        self._disabled_warning_printed = False
        self._geo = OfflineGeoIpResolver(os.getenv("WEBUI_V2_GEOIP_MMDB_PATH", ""))
        self._price_table = self._load_price_table()

    def _enabled(self) -> bool:
        return sqlite3 is not None

    def _warn_disabled(self) -> None:
        if self._disabled_warning_printed:
            return
        self._disabled_warning_printed = True
        print(f"[analytics] disabled: sqlite3 is unavailable ({_SQLITE_IMPORT_ERROR})")

    def _load_price_table(self) -> Dict[str, Dict[str, float]]:
        raw = os.getenv("WEBUI_V2_TOKEN_PRICE_JSON", "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {}
            table: Dict[str, Dict[str, float]] = {}
            for model, payload in parsed.items():
                if not isinstance(payload, dict):
                    continue
                table[str(model)] = {
                    "input_per_1k": _coerce_float(payload.get("input_per_1k")),
                    "output_per_1k": _coerce_float(payload.get("output_per_1k")),
                }
            return table
        except Exception:
            return {}

    @contextmanager
    def _conn(self):
        if sqlite3 is None:
            raise RuntimeError(f"sqlite3 is unavailable: {_SQLITE_IMPORT_ERROR}")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return
            if not self._enabled():
                self._warn_disabled()
                self._initialized = True
                return
            with self._conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        latency_ms INTEGER NOT NULL DEFAULT 0,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        usage_missing INTEGER NOT NULL DEFAULT 0,
                        model TEXT NOT NULL DEFAULT '',
                        owner_key TEXT NOT NULL DEFAULT '',
                        ip TEXT NOT NULL DEFAULT '',
                        country_code TEXT NOT NULL DEFAULT 'UNKNOWN',
                        region TEXT NOT NULL DEFAULT 'UNKNOWN',
                        city TEXT NOT NULL DEFAULT 'UNKNOWN',
                        estimated_cost_usd REAL NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS access_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        owner_key TEXT NOT NULL,
                        ip TEXT NOT NULL,
                        user_agent TEXT NOT NULL DEFAULT '',
                        country_code TEXT NOT NULL DEFAULT 'UNKNOWN',
                        region TEXT NOT NULL DEFAULT 'UNKNOWN',
                        city TEXT NOT NULL DEFAULT 'UNKNOWN'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ip_daily_rollup (
                        date TEXT NOT NULL,
                        country_code TEXT NOT NULL,
                        region TEXT NOT NULL,
                        uv INTEGER NOT NULL DEFAULT 0,
                        pv INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (date, country_code, region)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS token_daily_rollup (
                        date TEXT NOT NULL,
                        model TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        estimated_cost_usd REAL NOT NULL DEFAULT 0,
                        PRIMARY KEY (date, model, tool_name)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ip_daily_unique_owner (
                        date TEXT NOT NULL,
                        country_code TEXT NOT NULL,
                        region TEXT NOT NULL,
                        owner_key TEXT NOT NULL,
                        PRIMARY KEY (date, country_code, region, owner_key)
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(ts)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_model ON tool_calls(model)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_country ON tool_calls(country_code)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_events_ts ON access_events(ts)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_events_country ON access_events(country_code)")
            self._initialized = True

    def estimate_cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_key = str(model or "")
        if not model_key:
            return 0.0
        price = self._price_table.get(model_key)
        if not price:
            return 0.0
        return (input_tokens / 1000.0) * price.get("input_per_1k", 0.0) + (output_tokens / 1000.0) * price.get("output_per_1k", 0.0)

    def resolve_geo(self, ip: str) -> GeoInfo:
        return self._geo.resolve(ip)

    def record_access_event(self, *, ts: str, endpoint: str, owner_key: str, ip: str, user_agent: str) -> None:
        self.ensure_initialized()
        if not self._enabled():
            return
        stamp = ts or _utc_now_iso()
        geo = self.resolve_geo(ip)
        date = stamp[:10]
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO access_events (ts, endpoint, owner_key, ip, user_agent, country_code, region, city)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stamp, endpoint, owner_key, ip, user_agent, geo.country_code, geo.region, geo.city),
            )
            conn.execute(
                """
                INSERT INTO ip_daily_rollup (date, country_code, region, uv, pv)
                VALUES (?, ?, ?, 0, 1)
                ON CONFLICT(date, country_code, region)
                DO UPDATE SET pv = pv + 1
                """,
                (date, geo.country_code, geo.region),
            )
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO ip_daily_unique_owner (date, country_code, region, owner_key)
                VALUES (?, ?, ?, ?)
                """,
                (date, geo.country_code, geo.region, owner_key or "unknown"),
            ).rowcount
            if inserted:
                conn.execute(
                    """
                    UPDATE ip_daily_rollup
                    SET uv = uv + 1
                    WHERE date = ? AND country_code = ? AND region = ?
                    """,
                    (date, geo.country_code, geo.region),
                )

    def record_tool_call(
        self,
        *,
        ts: str,
        session_id: str,
        tool_name: str,
        status: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        usage_missing: bool,
        model: str,
        owner_key: str,
        ip: str,
    ) -> None:
        self.ensure_initialized()
        if not self._enabled():
            return
        stamp = ts or _utc_now_iso()
        geo = self.resolve_geo(ip)
        date = stamp[:10]
        estimated_cost = self.estimate_cost_usd(model, input_tokens, output_tokens)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls (
                    ts, session_id, tool_name, status, latency_ms,
                    input_tokens, output_tokens, total_tokens, usage_missing,
                    model, owner_key, ip, country_code, region, city, estimated_cost_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stamp,
                    session_id,
                    tool_name,
                    status,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    1 if usage_missing else 0,
                    model or "",
                    owner_key or "",
                    ip or "",
                    geo.country_code,
                    geo.region,
                    geo.city,
                    estimated_cost,
                ),
            )
            conn.execute(
                """
                INSERT INTO token_daily_rollup (
                    date, model, tool_name, input_tokens, output_tokens, total_tokens, estimated_cost_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, model, tool_name)
                DO UPDATE SET
                    input_tokens = input_tokens + excluded.input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    total_tokens = total_tokens + excluded.total_tokens,
                    estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd
                """,
                (date, model or "", tool_name, input_tokens, output_tokens, total_tokens, estimated_cost),
            )

    def _fetch_rows(self, sql: str, params: Iterable[Any]) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        if not self._enabled():
            return []
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def query_overview(self, from_iso: str, to_iso: str) -> Dict[str, Any]:
        if not self._enabled():
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "active_owners": 0,
                "unique_ips": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        rows = self._fetch_rows(
            """
            SELECT
              COUNT(*) AS total_calls,
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_calls,
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_calls,
              SUM(input_tokens) AS input_tokens,
              SUM(output_tokens) AS output_tokens,
              SUM(total_tokens) AS total_tokens,
              SUM(estimated_cost_usd) AS estimated_cost_usd
            FROM tool_calls
            WHERE ts >= ? AND ts <= ?
            """,
            (from_iso, to_iso),
        )[0]
        owners = self._fetch_rows(
            "SELECT COUNT(DISTINCT owner_key) AS active_owners, COUNT(DISTINCT ip) AS unique_ips FROM access_events WHERE ts >= ? AND ts <= ?",
            (from_iso, to_iso),
        )[0]
        total_calls = _coerce_int(rows.get("total_calls"))
        successful_calls = _coerce_int(rows.get("successful_calls"))
        success_rate = 0.0 if total_calls <= 0 else (successful_calls / total_calls) * 100.0
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": _coerce_int(rows.get("failed_calls")),
            "success_rate": success_rate,
            "active_owners": _coerce_int(owners.get("active_owners")),
            "unique_ips": _coerce_int(owners.get("unique_ips")),
            "input_tokens": _coerce_int(rows.get("input_tokens")),
            "output_tokens": _coerce_int(rows.get("output_tokens")),
            "total_tokens": _coerce_int(rows.get("total_tokens")),
            "estimated_cost_usd": _coerce_float(rows.get("estimated_cost_usd")),
        }

    def query_tool_calls(self, from_iso: str, to_iso: str, group_by: str) -> List[Dict[str, Any]]:
        if group_by == "tool":
            sql = """
            SELECT
              tool_name AS bucket,
              tool_name,
              COUNT(*) AS calls,
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_calls,
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_calls,
              AVG(latency_ms) AS avg_latency_ms,
              SUM(total_tokens) AS total_tokens
            FROM tool_calls
            WHERE ts >= ? AND ts <= ?
            GROUP BY tool_name
            ORDER BY calls DESC
            """
        elif group_by == "hour":
            sql = """
            SELECT
              substr(ts, 1, 13) AS bucket,
              '' AS tool_name,
              COUNT(*) AS calls,
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_calls,
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_calls,
              AVG(latency_ms) AS avg_latency_ms,
              SUM(total_tokens) AS total_tokens
            FROM tool_calls
            WHERE ts >= ? AND ts <= ?
            GROUP BY substr(ts, 1, 13)
            ORDER BY bucket ASC
            """
        else:
            sql = """
            SELECT
              substr(ts, 1, 10) AS bucket,
              '' AS tool_name,
              COUNT(*) AS calls,
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_calls,
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_calls,
              AVG(latency_ms) AS avg_latency_ms,
              SUM(total_tokens) AS total_tokens
            FROM tool_calls
            WHERE ts >= ? AND ts <= ?
            GROUP BY substr(ts, 1, 10)
            ORDER BY bucket ASC
            """
        rows = self._fetch_rows(sql, (from_iso, to_iso))
        for row in rows:
            row["calls"] = _coerce_int(row.get("calls"))
            row["successful_calls"] = _coerce_int(row.get("successful_calls"))
            row["failed_calls"] = _coerce_int(row.get("failed_calls"))
            row["avg_latency_ms"] = int(_coerce_float(row.get("avg_latency_ms")))
            row["total_tokens"] = _coerce_int(row.get("total_tokens"))
        return rows

    def query_ip_distribution(self, from_iso: str, to_iso: str, level: str) -> List[Dict[str, Any]]:
        if level == "region":
            group = "country_code, region"
            order = "pv DESC"
        else:
            group = "country_code"
            order = "pv DESC"
        sql = f"""
        SELECT country_code, COALESCE(region, 'UNKNOWN') AS region, COUNT(*) AS pv, COUNT(DISTINCT owner_key) AS uv
        FROM access_events
        WHERE ts >= ? AND ts <= ?
        GROUP BY {group}
        ORDER BY {order}
        """
        rows = self._fetch_rows(sql, (from_iso, to_iso))
        for row in rows:
            row["pv"] = _coerce_int(row.get("pv"))
            row["uv"] = _coerce_int(row.get("uv"))
        return rows

    def query_token_usage(self, from_iso: str, to_iso: str, group_by: str) -> List[Dict[str, Any]]:
        if group_by == "model":
            group = "model"
            bucket = "model"
        elif group_by == "tool":
            group = "tool_name"
            bucket = "tool_name"
        else:
            group = "substr(ts, 1, 10)"
            bucket = "substr(ts, 1, 10)"
        sql = f"""
        SELECT
          {bucket} AS bucket,
          COALESCE(model, '') AS model,
          COALESCE(tool_name, '') AS tool_name,
          SUM(input_tokens) AS input_tokens,
          SUM(output_tokens) AS output_tokens,
          SUM(total_tokens) AS total_tokens,
          SUM(estimated_cost_usd) AS estimated_cost_usd
        FROM tool_calls
        WHERE ts >= ? AND ts <= ?
        GROUP BY {group}
        ORDER BY total_tokens DESC
        """
        rows = self._fetch_rows(sql, (from_iso, to_iso))
        for row in rows:
            row["input_tokens"] = _coerce_int(row.get("input_tokens"))
            row["output_tokens"] = _coerce_int(row.get("output_tokens"))
            row["total_tokens"] = _coerce_int(row.get("total_tokens"))
            row["estimated_cost_usd"] = _coerce_float(row.get("estimated_cost_usd"))
        return rows

    def query_map(self, from_iso: str, to_iso: str) -> List[Dict[str, Any]]:
        rows = self._fetch_rows(
            """
            SELECT country_code, COUNT(*) AS pv, COUNT(DISTINCT owner_key) AS uv
            FROM access_events
            WHERE ts >= ? AND ts <= ?
            GROUP BY country_code
            ORDER BY pv DESC
            """,
            (from_iso, to_iso),
        )
        max_pv = max([_coerce_int(row.get("pv")) for row in rows] or [1])
        result: List[Dict[str, Any]] = []
        for row in rows:
            pv = _coerce_int(row.get("pv"))
            result.append(
                {
                    "country_code": row.get("country_code") or "UNKNOWN",
                    "pv": pv,
                    "uv": _coerce_int(row.get("uv")),
                    "intensity": pv / max_pv if max_pv > 0 else 0,
                }
            )
        return result


analytics_store = AnalyticsStore()


def normalize_time_range(from_value: Optional[str], to_value: Optional[str]) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    end = now if not to_value else _parse_iso(to_value)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    if from_value:
        start = _parse_iso(from_value)
    else:
        start = start - timedelta(days=7)
    if start > end:
        start, end = end, start
    return start.isoformat(), end.isoformat()
