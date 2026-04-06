import os
import tempfile
import unittest
from pathlib import Path

from src.web_v2.analytics_store import AnalyticsStore, normalize_time_range


class AnalyticsStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev_db_path = os.getenv("WEBUI_V2_ANALYTICS_DB_PATH")
        self.prev_price_json = os.getenv("WEBUI_V2_TOKEN_PRICE_JSON")
        os.environ["WEBUI_V2_ANALYTICS_DB_PATH"] = str(Path(self.tmp.name) / "analytics.db")
        os.environ["WEBUI_V2_TOKEN_PRICE_JSON"] = '{"test-model":{"input_per_1k":0.001,"output_per_1k":0.002}}'
        self.store = AnalyticsStore()
        self.store.ensure_initialized()

    def tearDown(self):
        if self.prev_db_path is None:
            os.environ.pop("WEBUI_V2_ANALYTICS_DB_PATH", None)
        else:
            os.environ["WEBUI_V2_ANALYTICS_DB_PATH"] = self.prev_db_path
        if self.prev_price_json is None:
            os.environ.pop("WEBUI_V2_TOKEN_PRICE_JSON", None)
        else:
            os.environ["WEBUI_V2_TOKEN_PRICE_JSON"] = self.prev_price_json
        self.tmp.cleanup()

    def test_record_and_query_overview(self):
        self.store.record_access_event(
            ts="2026-04-05T08:00:00+00:00",
            endpoint="/api/chat/sessions:create",
            owner_key="owner-a",
            ip="127.0.0.1",
            user_agent="ua",
        )
        self.store.record_access_event(
            ts="2026-04-05T09:00:00+00:00",
            endpoint="/api/chat/quota",
            owner_key="owner-b",
            ip="127.0.0.2",
            user_agent="ua",
        )
        self.store.record_tool_call(
            ts="2026-04-05T10:00:00+00:00",
            session_id="s1",
            tool_name="python_repl",
            status="success",
            latency_ms=1200,
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            usage_missing=False,
            model="test-model",
            owner_key="owner-a",
            ip="127.0.0.1",
        )
        self.store.record_tool_call(
            ts="2026-04-05T10:05:00+00:00",
            session_id="s1",
            tool_name="search_arxiv",
            status="failed",
            latency_ms=300,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            usage_missing=True,
            model="",
            owner_key="owner-a",
            ip="127.0.0.1",
        )

        overview = self.store.query_overview("2026-04-05T00:00:00+00:00", "2026-04-05T23:59:59+00:00")
        self.assertEqual(overview["total_calls"], 2)
        self.assertEqual(overview["successful_calls"], 1)
        self.assertEqual(overview["failed_calls"], 1)
        self.assertEqual(overview["active_owners"], 2)
        self.assertEqual(overview["unique_ips"], 2)
        self.assertEqual(overview["total_tokens"], 1500)
        self.assertGreater(overview["estimated_cost_usd"], 0)

    def test_query_dimensions(self):
        self.store.record_access_event(
            ts="2026-04-05T08:00:00+00:00",
            endpoint="/api/chat/sessions:list",
            owner_key="owner-a",
            ip="127.0.0.1",
            user_agent="ua",
        )
        self.store.record_tool_call(
            ts="2026-04-05T08:10:00+00:00",
            session_id="s1",
            tool_name="tool_a",
            status="success",
            latency_ms=100,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            usage_missing=False,
            model="test-model",
            owner_key="owner-a",
            ip="127.0.0.1",
        )
        by_tool = self.store.query_tool_calls("2026-04-05T00:00:00+00:00", "2026-04-05T23:59:59+00:00", "tool")
        self.assertEqual(len(by_tool), 1)
        self.assertEqual(by_tool[0]["tool_name"], "tool_a")
        token_by_day = self.store.query_token_usage("2026-04-05T00:00:00+00:00", "2026-04-05T23:59:59+00:00", "day")
        self.assertEqual(len(token_by_day), 1)
        ip_dist = self.store.query_ip_distribution("2026-04-05T00:00:00+00:00", "2026-04-05T23:59:59+00:00", "country")
        self.assertEqual(len(ip_dist), 1)
        map_rows = self.store.query_map("2026-04-05T00:00:00+00:00", "2026-04-05T23:59:59+00:00")
        self.assertEqual(len(map_rows), 1)

    def test_normalize_time_range(self):
        start, end = normalize_time_range("2026-04-01T00:00:00+00:00", "2026-04-05T00:00:00+00:00")
        self.assertTrue(start <= end)


if __name__ == "__main__":
    unittest.main()
