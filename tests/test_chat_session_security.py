import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.web_v2 import chat_api


class _FakeRequest:
    def __init__(self, ip: str, user_agent: str = "ua", origin: str = "http://localhost"):
        self.headers = {
            "x-forwarded-for": ip,
            "user-agent": user_agent,
            "origin": origin,
        }
        self.client = SimpleNamespace(host=ip)


class ChatSessionSecurityTests(unittest.TestCase):
    def setUp(self):
        self._mode_patcher = patch.object(chat_api, "_runtime_mode", return_value="online")
        self._mode_patcher.start()

    def tearDown(self):
        self._mode_patcher.stop()

    def test_issue_and_verify_token(self):
        state = {}
        req = _FakeRequest("10.0.0.1")
        token, _ = chat_api._issue_session_access_token(state, req)  # noqa: SLF001
        req_with_token = _FakeRequest("10.0.0.1")
        req_with_token.headers["x-session-access-token"] = token
        chat_api._assert_session_access(state, req_with_token)  # noqa: SLF001

    def test_reject_owner_mismatch(self):
        state = {}
        req = _FakeRequest("10.0.0.1")
        token, _ = chat_api._issue_session_access_token(state, req)  # noqa: SLF001
        req_other = _FakeRequest("10.0.0.2")
        req_other.headers["x-session-access-token"] = token
        with self.assertRaises(Exception):
            chat_api._assert_session_access(state, req_other)  # noqa: SLF001

    def test_reject_invalid_token(self):
        state = {}
        req = _FakeRequest("10.0.0.1")
        chat_api._issue_session_access_token(state, req)  # noqa: SLF001
        req_with_invalid_token = _FakeRequest("10.0.0.1")
        req_with_invalid_token.headers["x-session-access-token"] = "invalid-token"
        with self.assertRaises(Exception):
            chat_api._assert_session_access(state, req_with_invalid_token)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
