"""
Runner de backtesting com Backtrader para XAUUSD (diário e horário).

Gera métricas (Sharpe, drawdown, retorno acumulado), curvas de equity/drawdown
e relatórios em HTML/PDF. Suporta múltiplas estratégias (SMA, RSI, MACD).
"""
from __future__ import annotations

import io
import os
import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

from .backtrader_strategies import SmaCrossStrategy, RsiStrategy, MacdStrategy


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class BacktestResult:
    strategy_name: str
    timeframe: str
    sharpe: float
    max_drawdown_pct: float
    total_return_pct: float
    trade_count: int
    equity_curve: List[float]
    drawdown_curve: List[float]


class PandasDataXAU(bt.feeds.PandasData):
    params = (
        ('datetime', 'timestamp'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
    )


class BacktraderRunner:
    """Executa backtesting com Backtrader e produz relatórios."""

    def __init__(self, initial_cash: float = 10000.0, commission: float = 0.001) -> None:
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        os.makedirs('results', exist_ok=True)

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if 'timestamp' not in df.columns:
            raise ValueError("CSV precisa conter coluna 'timestamp'")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"CSV não possui coluna obrigatória: {col}")
        return df

    def _setup_cerebro(self, df: pd.DataFrame, strategy_cls: type[bt.Strategy], strategy_params: Dict | None = None) -> Tuple[bt.Cerebro, bt.Strategy]:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        data = PandasDataXAU(dataname=df)
        cerebro.adddata(data)
        if strategy_params:
            cerebro.addstrategy(strategy_cls, **strategy_params)
        else:
            cerebro.addstrategy(strategy_cls)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='ddown')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        return cerebro

    def run_once(self, csv_path: str, strategy_name: str, timeframe: str, strategy_params: Dict | None = None) -> BacktestResult:
        df = self._load_csv(csv_path)
        strategy_map = {
            'sma': SmaCrossStrategy,
            'rsi': RsiStrategy,
            'macd': MacdStrategy,
        }
        if strategy_name not in strategy_map:
            raise ValueError(f"Estratégia desconhecida: {strategy_name}")
        cerebro = self._setup_cerebro(df, strategy_map[strategy_name], strategy_params)
        strat_runs = cerebro.run()
        strat = strat_runs[0]
        analyzers = strat.analyzers
        sharpe = float(analyzers.sharpe.get_analysis().get('sharperatio', 0.0) or 0.0)
        dd = analyzers.ddown.get_analysis()
        max_dd_pct = float(dd.get('max', {}).get('drawdown', 0.0) or 0.0)
        trets = analyzers.time_return.get_analysis()
        returns = np.array(list(trets.values()), dtype=float) if trets else np.array([], dtype=float)
        total_return_pct = float((np.prod(1.0 + returns) - 1.0) * 100.0) if returns.size > 0 else 0.0
        trades = analyzers.trades.get_analysis()
        trade_count = int(trades.get('total').get('closed', 0)) if trades and trades.get('total') else 0

        equity_curve = getattr(strat, 'equity_series', []) or list(self.initial_cash * np.cumprod(1.0 + returns))
        drawdown_curve = getattr(strat, 'drawdown_series', [])

        result = BacktestResult(
            strategy_name=strategy_name,
            timeframe=timeframe,
            sharpe=sharpe,
            max_drawdown_pct=max_dd_pct,
            total_return_pct=total_return_pct,
            trade_count=trade_count,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
        )
        return result

    def plot_curves(self, result: BacktestResult, out_prefix: str) -> Tuple[str, str]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(result.equity_curve, color='blue')
        axes[0].axhline(y=self.initial_cash, color='gray', linestyle='--')
        axes[0].set_title(f"Equity Curve — {result.strategy_name.upper()} ({result.timeframe})")
        axes[0].set_ylabel('Valor ($)')
        axes[0].grid(True, alpha=0.3)

        if result.drawdown_curve:
            axes[1].plot(result.drawdown_curve, color='red')
        else:
            eq = np.array(result.equity_curve, dtype=float)
            if eq.size > 0:
                run_max = np.maximum.accumulate(eq)
                dd = (eq - run_max) / run_max * 100.0
                axes[1].plot(dd, color='red')
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('%')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        img_path = f"results/{out_prefix}_{result.strategy_name}_{result.timeframe}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Base64 para HTML
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        return img_path, f"data:image/png;base64,{b64}"

    def generate_html_report(self, results: List[BacktestResult], title: str, image_data_uris: List[str], out_path: str) -> str:
        rows = ''.join([
            f"<tr><td>{r.strategy_name}</td><td>{r.timeframe}</td><td>{r.sharpe:.3f}</td><td>{r.max_drawdown_pct:.2f}%</td><td>{r.total_return_pct:.2f}%</td><td>{r.trade_count}</td></tr>"
            for r in results
        ])
        imgs = ''.join([f"<img src='{uri}' style='width: 100%; margin-bottom: 12px;'/>" for uri in image_data_uris])
        html = f"""
<!doctype html>
<html lang='pt-BR'>
<head>
  <meta charset='utf-8'/>
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <h2>Resumo de Métricas</h2>
  <table>
    <thead><tr><th>Estratégia</th><th>Timeframe</th><th>Sharpe</th><th>Max DD</th><th>Retorno</th><th>Trades</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <h2>Gráficos</h2>
  {imgs}
</body>
</html>
"""
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return out_path

    def generate_pdf_report(self, results: List[BacktestResult], title: str, image_paths: List[str], out_path: str) -> str:
        doc = SimpleDocTemplate(out_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story: List = []
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))

        data = [["Estratégia", "Timeframe", "Sharpe", "Max DD", "Retorno", "Trades"]]
        for r in results:
            data.append([r.strategy_name.upper(), r.timeframe, f"{r.sharpe:.3f}", f"{r.max_drawdown_pct:.2f}%", f"{r.total_return_pct:.2f}%", str(r.trade_count)])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        for path in image_paths:
            story.append(Image(path, width=500, height=300))
            story.append(Spacer(1, 12))

        doc.build(story)
        return out_path

    def run_batch(self, inputs: List[Tuple[str, str, Dict | None]]) -> Tuple[List[BacktestResult], List[str], List[str]]:
        results: List[BacktestResult] = []
        image_paths: List[str] = []
        image_uris: List[str] = []
        for csv_path, strategy_name, params in inputs:
            timeframe = '1d' if '1d' in os.path.basename(csv_path) else ('1h' if '1h' in os.path.basename(csv_path) else 'unknown')
            try:
                res = self.run_once(csv_path, strategy_name=strategy_name, timeframe=timeframe, strategy_params=params)
                results.append(res)
                img_path, img_uri = self.plot_curves(res, out_prefix='xauusd')
                image_paths.append(img_path)
                image_uris.append(img_uri)
                logger.info(f"{strategy_name.upper()} {timeframe}: Sharpe={res.sharpe:.3f} DD={res.max_drawdown_pct:.2f}% Ret={res.total_return_pct:.2f}%")
            except Exception as e:
                logger.error(f"Falha ao rodar {strategy_name} em {csv_path}: {e}")
        return results, image_paths, image_uris