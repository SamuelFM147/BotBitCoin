"""
Validação completa de estratégias para XAUUSD (Ouro/Dólar).

Inclui:
- Backtesting com Backtrader em dados diários e horários
- Teste de estratégias SMA, RSI, MACD
- Avaliação por condições de mercado (tendência/volatilidade)
- Walk-forward testing
- Relatórios HTML e PDF automatizados
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .backtrader_runner import BacktraderRunner, BacktestResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def segment_market_conditions(df: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
    close = df['close'].astype(float).values
    returns = np.diff(close) / close[:-1]
    ret_series = np.concatenate([[0.0], returns])
    roll_mean = pd.Series(ret_series).rolling(window=window, min_periods=1).mean().values
    roll_vol = pd.Series(ret_series).rolling(window=window, min_periods=1).std().values
    trend_up_mask = roll_mean > 0
    trend_down_mask = roll_mean <= 0
    vol_high_mask = roll_vol > np.nanmedian(roll_vol)
    vol_low_mask = roll_vol <= np.nanmedian(roll_vol)
    conds = {
        'trend_up': df.loc[trend_up_mask],
        'trend_down': df.loc[trend_down_mask],
        'vol_high': df.loc[vol_high_mask],
        'vol_low': df.loc[vol_low_mask],
    }
    return conds


def walk_forward(df: pd.DataFrame, runner: BacktraderRunner, strategy_name: str, window_size: int = 250, step: int = 125) -> List[BacktestResult]:
    results: List[BacktestResult] = []
    n = len(df)
    start = 0
    idx = 0
    while (start + window_size) <= n:
        slice_df = df.iloc[start:start + window_size].reset_index(drop=True)
        # Salva temporário para reuso do runner
        tmp_path = f"python/data/_tmp_xau_wf_{strategy_name}_{idx}.csv"
        slice_df.to_csv(tmp_path, index=False)
        res = runner.run_once(tmp_path, strategy_name=strategy_name, timeframe='wf', strategy_params=None)
        results.append(res)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        start += step
        idx += 1
    return results


def validate_xauusd(daily_csv: str, hourly_csv: str | None = None, initial_cash: float = 10000.0, commission: float = 0.001) -> Dict[str, str]:
    runner = BacktraderRunner(initial_cash=initial_cash, commission=commission)
    inputs: List[Tuple[str, str, Dict | None]] = []
    inputs.append((daily_csv, 'sma', {'period_short': 10, 'period_long': 30}))
    inputs.append((daily_csv, 'rsi', {'period': 14, 'rsi_low': 30, 'rsi_high': 70}))
    inputs.append((daily_csv, 'macd', {'fast': 12, 'slow': 26, 'signal': 9}))
    if hourly_csv and os.path.exists(hourly_csv):
        inputs.append((hourly_csv, 'sma', {'period_short': 20, 'period_long': 50}))
        inputs.append((hourly_csv, 'rsi', {'period': 14, 'rsi_low': 30, 'rsi_high': 70}))
        inputs.append((hourly_csv, 'macd', {'fast': 12, 'slow': 26, 'signal': 9}))

    results, image_paths, image_uris = runner.run_batch(inputs)

    # Avaliação por condições de mercado no diário
    try:
        df_daily = _load_csv(daily_csv)
        conds = segment_market_conditions(df_daily, window=20)
        for name, subdf in conds.items():
            if len(subdf) < 50:
                continue
            tmp = f"python/data/_tmp_xau_{name}.csv"
            subdf.to_csv(tmp, index=False)
            for strat in ['sma', 'rsi', 'macd']:
                _ = runner.run_once(tmp, strategy_name=strat, timeframe=name, strategy_params=None)
            try:
                os.remove(tmp)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Avaliação por condições falhou: {e}")

    # Walk-forward no diário (janelas de ~1 ano)
    try:
        wf_results_all: List[BacktestResult] = []
        for strat in ['sma', 'rsi', 'macd']:
            wf_strat_results = walk_forward(_load_csv(daily_csv), runner, strategy_name=strat, window_size=252, step=126)
            wf_results_all.extend(wf_strat_results)
    except Exception as e:
        logger.warning(f"Walk-forward falhou: {e}")

    # Relatórios
    html_title = "Validação XAUUSD — Estratégias Clássicas"
    html_path = "results/xauusd_validation_report.html"
    pdf_path = "results/xauusd_validation_report.pdf"
    try:
        runner.generate_html_report(results, title=html_title, image_data_uris=image_uris, out_path=html_path)
    except Exception as e:
        logger.error(f"Falha ao gerar HTML: {e}")
    try:
        runner.generate_pdf_report(results, title=html_title, image_paths=image_paths, out_path=pdf_path)
    except Exception as e:
        logger.error(f"Falha ao gerar PDF: {e}")

    return {
        'html_report': html_path,
        'pdf_report': pdf_path,
    }