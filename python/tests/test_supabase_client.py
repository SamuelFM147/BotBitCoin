import os
import json
from unittest import mock

import pytest

from integrations.supabase_client import SupabaseEdgeClient


def test_supabase_client_constructs_function_url_correctly():
    os.environ["VITE_SUPABASE_URL"] = "https://example.supabase.co"
    os.environ["VITE_SUPABASE_PUBLISHABLE_KEY"] = "anon-key"

    client = SupabaseEdgeClient()
    assert client.function_url == "https://example.functions.supabase.co/rl-training"


def test_save_episode_posts_payload_and_parses_response():
    os.environ["VITE_SUPABASE_URL"] = "https://example.supabase.co"
    os.environ["VITE_SUPABASE_PUBLISHABLE_KEY"] = "anon-key"
    client = SupabaseEdgeClient()

    with mock.patch("requests.post") as post:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "episode": {"id": 123}}
        post.return_value = mock_resp

        resp = client.save_episode(
            agent_id="DQN-v2.1",
            episode_number=1,
            total_reward=10.5,
            avg_loss=0.123,
            epsilon=0.1,
            actions_taken=42,
            duration_seconds=3.2,
        )
        assert resp["success"] is True
        assert resp["episode"]["id"] == 123

        # Verify URL and headers
        called_url = post.call_args[0][0]
        called_headers = post.call_args[1]["headers"]
        called_data = json.loads(post.call_args[1]["data"])
        assert called_url == "https://example.functions.supabase.co/rl-training"
        assert called_headers["Authorization"].startswith("Bearer ")
        assert called_data["action"] == "save_episode"


def test_save_trades_batch_maps_fields():
    os.environ["VITE_SUPABASE_URL"] = "https://example.supabase.co"
    os.environ["VITE_SUPABASE_PUBLISHABLE_KEY"] = "anon-key"
    client = SupabaseEdgeClient()

    trades = [
        {"action": "buy", "executed_price": 100.0, "amount": 0.1},
        {"action": "sell", "price": 110.0, "amount": 0.05, "profit_loss": 0.25},
    ]

    with mock.patch("requests.post") as post:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "trade": {"id": 999}}
        post.return_value = mock_resp

        resps = client.save_trades_batch(
            agent_id="DQN-v2.1", episode_id=123, trades=trades, default_confidence=None
        )
        assert len(resps) == 2

        # First call should use executed_price
        first_payload = json.loads(post.call_args_list[0][1]["data"])
        assert first_payload["action"] == "save_trade"
        assert first_payload["data"]["trade_type"] == "buy"
        assert first_payload["data"]["price"] == 100.0
        assert first_payload["data"]["amount"] == 0.1

        # Second call should use price field
        second_payload = json.loads(post.call_args_list[1][1]["data"])
        assert second_payload["data"]["trade_type"] == "sell"
        assert second_payload["data"]["price"] == 110.0
        assert second_payload["data"]["amount"] == 0.05
        assert second_payload["data"]["profit_loss"] == 0.25