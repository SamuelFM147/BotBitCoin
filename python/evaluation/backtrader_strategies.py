"""
Estratégias Backtrader clássicas para validação em XAUUSD.

Contém implementações de SMA Crossover, RSI e MACD.
"""
from __future__ import annotations

from typing import Any

import backtrader as bt


class BaseRecordingStrategy(bt.Strategy):
    """Estratégia base que grava equity e drawdown por passo.

    Útil para geração de curvas de equity e análise de drawdown.
    """

    params = dict()

    def __init__(self) -> None:
        self.equity_series: list[float] = []
        self.drawdown_series: list[float] = []
        self._max_value: float = 0.0

    def next(self) -> None:
        val = float(self.broker.getvalue())
        self.equity_series.append(val)
        self._max_value = max(self._max_value, val)
        dd = (val - self._max_value) / self._max_value * 100 if self._max_value > 0 else 0.0
        self.drawdown_series.append(dd)


class SmaCrossStrategy(BaseRecordingStrategy):
    """SMA Crossover: compra quando SMA curta cruza acima da longa, vende no contrário."""

    params = dict(period_short=10, period_long=30, stake=1.0)

    def __init__(self) -> None:
        super().__init__()
        sma_short = bt.ind.SMA(self.data.close, period=int(self.params.period_short))
        sma_long = bt.ind.SMA(self.data.close, period=int(self.params.period_long))
        self.crossover = bt.ind.CrossOver(sma_short, sma_long)

    def next(self) -> None:
        super().next()
        if not self.position:
            if self.crossover > 0:
                size = max(1, int(self.broker.getcash() / float(self.data.close[0]) * float(self.params.stake)))
                self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()


class RsiStrategy(BaseRecordingStrategy):
    """RSI: compra em oversold e vende em overbought."""

    params = dict(period=14, rsi_low=30, rsi_high=70, stake=1.0)

    def __init__(self) -> None:
        super().__init__()
        self.rsi = bt.ind.RSI(self.data.close, period=int(self.params.period))

    def next(self) -> None:
        super().next()
        if not self.position and float(self.rsi[0]) < float(self.params.rsi_low):
            size = max(1, int(self.broker.getcash() / float(self.data.close[0]) * float(self.params.stake)))
            self.buy(size=size)
        elif self.position and float(self.rsi[0]) > float(self.params.rsi_high):
            self.close()


class MacdStrategy(BaseRecordingStrategy):
    """MACD: segue cruzamento de linha MACD com sinal."""

    params = dict(fast=12, slow=26, signal=9, stake=1.0)

    def __init__(self) -> None:
        super().__init__()
        macd = bt.ind.MACD(self.data.close, period_me1=int(self.params.fast), period_me2=int(self.params.slow), period_signal=int(self.params.signal))
        self.signal = bt.ind.CrossOver(macd.macd, macd.signal)

    def next(self) -> None:
        super().next()
        if not self.position and self.signal > 0:
            size = max(1, int(self.broker.getcash() / float(self.data.close[0]) * float(self.params.stake)))
            self.buy(size=size)
        elif self.position and self.signal < 0:
            self.close()