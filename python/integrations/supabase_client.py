"""
Supabase Edge Function client for Python pipeline

Provides simple, robust helpers to persist training episodes and trades.
Reads existing Vite env vars from .env when available and derives the
Edge Function URL from the Supabase project URL.
"""
from __future__ import annotations

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union

import requests

try:
    # Load .env if python-dotenv is available; otherwise rely on OS env
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SupabaseEdgeClient:
    """Client for calling Supabase Edge Function `rl-training`.

    It expects the environment variables used by the frontend build:
    - `VITE_SUPABASE_URL` (e.g., https://<project>.supabase.co)
    - `VITE_SUPABASE_PUBLISHABLE_KEY` (anon key)

    The function base URL is derived as:
    https://<project>.functions.supabase.co/rl-training
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        function_path: str = "rl-training",
        timeout_seconds: int = 10,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.75,
    ):
        supabase_url = supabase_url or os.getenv("VITE_SUPABASE_URL")
        # Preferir service role (quando disponível) para chamadas do pipeline Python
        supabase_key = (
            supabase_anon_key
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY")
        )

        if not supabase_url:
            raise ValueError("Supabase URL not found. Set VITE_SUPABASE_URL or pass supabase_url.")
        if not supabase_key:
            raise ValueError(
                "Supabase key not found. Set SUPABASE_SERVICE_ROLE_KEY or VITE_SUPABASE_PUBLISHABLE_KEY, ou passe supabase_anon_key."
            )

        # Derive function base from Supabase URL
        # https://<project>.supabase.co -> https://<project>.functions.supabase.co
        self.function_base = supabase_url.replace(".supabase.co", ".functions.supabase.co")
        self.function_url = f"{self.function_base}/{function_path}"

        self.timeout_seconds = timeout_seconds

        # Controla retries via variáveis de ambiente
        disable_retries_env = (os.getenv("SUPABASE_DISABLE_RETRIES") or "").strip().lower()
        max_retries_env = (os.getenv("SUPABASE_MAX_RETRIES") or "").strip()

        if disable_retries_env in {"true", "1", "yes", "on"}:
            # Tenta apenas uma vez (sem backoff) quando desativado
            self.max_retries = 1
            self.retry_backoff_seconds = 0.0
            logger.info("Supabase retries desativados via SUPABASE_DISABLE_RETRIES")
        elif max_retries_env.isdigit():
            # Ajusta retries explicitamente, garantindo >= 1
            self.max_retries = max(1, int(max_retries_env))
            self.retry_backoff_seconds = retry_backoff_seconds
            logger.info(f"Supabase retries ajustados via SUPABASE_MAX_RETRIES={self.max_retries}")
        else:
            self.max_retries = max_retries
            self.retry_backoff_seconds = retry_backoff_seconds
        self.headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Apikey": supabase_key,
        }

        logger.info(f"Supabase Edge Function configured: {self.function_url}")

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST helper with simple retries and JSON response parsing."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.function_url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout_seconds)
                if resp.status_code >= 200 and resp.status_code < 300:
                    return resp.json()
                else:
                    # Attempt to parse error message
                    try:
                        err_json = resp.json()
                        msg = err_json.get("error") or str(err_json)
                    except Exception:
                        msg = resp.text
                    raise RuntimeError(f"Supabase function error {resp.status_code}: {msg}")
            except Exception as e:
                last_err = e
                logger.warning(f"Supabase POST failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds)
        # If we get here, all attempts failed
        if last_err:
            raise last_err
        raise RuntimeError("Supabase POST failed for unknown reasons")

    # Public API
    def save_episode(
        self,
        agent_id: str,
        episode_number: int,
        total_reward: float,
        avg_loss: float,
        epsilon: float,
        actions_taken: int,
        duration_seconds: float,
    ) -> Dict[str, Any]:
        """Persist episode summary into `training_episodes`.

        Returns the JSON with the inserted `episode` which contains its `id`.
        """
        payload = {
            "action": "save_episode",
            "data": {
                "agent_id": agent_id,
                "episode_number": episode_number,
                "total_reward": float(total_reward),
                "avg_loss": float(avg_loss),
                "epsilon": float(epsilon),
                "actions_taken": int(actions_taken),
                "duration_seconds": float(duration_seconds),
            },
        }
        return self._post(payload)

    def save_trade(
        self,
        agent_id: str,
        episode_id: Union[str, int],
        trade_type: str,
        price: float,
        amount: float,
        profit_loss: float = 0.0,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Persist a single trade into `trades` table."""
        # Normalize trade_type to schema-supported values
        trade_type_norm = str(trade_type).lower()
        if trade_type_norm not in {"buy", "sell", "hold"}:
            trade_type_norm = "sell" if trade_type_norm in {"auto_close", "close", "exit"} else "hold"

        payload = {
            "action": "save_trade",
            "data": {
                "agent_id": agent_id,
                "episode_id": str(episode_id),
                "trade_type": trade_type_norm,
                "price": float(price),
                "amount": float(amount),
                "profit_loss": float(profit_loss or 0.0),
                "confidence": None if confidence is None else float(confidence),
            },
        }
        return self._post(payload)

    def save_trades_batch(
        self,
        agent_id: str,
        episode_id: Union[str, int],
        trades: List[Dict[str, Any]],
        default_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Persist a list of environment trades with mapping to Supabase fields.

        Expects each trade dict to include keys produced by BitcoinTradingEnv like:
        - action: 'buy' | 'sell' | 'auto_close'
        - executed_price or price
        - amount
        - profit_loss (optional)
        """
        results: List[Dict[str, Any]] = []
        for t in trades:
            trade_type = t.get("action", "")
            price = float(t.get("executed_price", t.get("price", 0.0)))
            amount = float(t.get("amount", 0.0))
            profit_loss = float(t.get("profit_loss", 0.0))
            res = self.save_trade(
                agent_id=agent_id,
                episode_id=episode_id,
                trade_type=trade_type,
                price=price,
                amount=amount,
                profit_loss=profit_loss,
                confidence=default_confidence,
            )
            results.append(res)
        return results