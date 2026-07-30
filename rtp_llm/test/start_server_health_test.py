from unittest import TestCase, main
from unittest.mock import Mock, patch

from rtp_llm.start_server import check_server_health


class StartServerHealthTest(TestCase):
    @patch("rtp_llm.start_server.requests.get")
    def test_accepts_supported_healthy_responses(self, mock_get):
        for health in ("ok", {"status": "ok"}):
            with self.subTest(health=health):
                response = Mock(status_code=200, text=str(health))
                response.json.return_value = health
                mock_get.return_value = response

                self.assertTrue(check_server_health(8088))
                mock_get.assert_called_with("http://localhost:8088/health", timeout=60)

    @patch("rtp_llm.start_server.requests.get")
    def test_rejects_non_healthy_responses(self, mock_get):
        for health in ("starting", {"status": "starting"}, {}, None):
            with self.subTest(health=health):
                response = Mock(status_code=200, text=str(health))
                response.json.return_value = health
                mock_get.return_value = response

                self.assertFalse(check_server_health(8088))

    @patch("rtp_llm.start_server.requests.get")
    def test_rejects_non_success_status_without_parsing_body(self, mock_get):
        response = Mock(status_code=503, text='"ok"')
        mock_get.return_value = response

        self.assertFalse(check_server_health(8088))
        response.json.assert_not_called()

    @patch("rtp_llm.start_server.requests.get")
    def test_returns_false_when_response_is_not_json(self, mock_get):
        response = Mock(status_code=200, text="not-json")
        response.json.side_effect = ValueError("invalid JSON")
        mock_get.return_value = response

        self.assertFalse(check_server_health(8088))

    @patch("rtp_llm.start_server.requests.get")
    def test_returns_false_when_request_fails(self, mock_get):
        mock_get.side_effect = RuntimeError("request failed")

        self.assertFalse(check_server_health(8088))


if __name__ == "__main__":
    main()
