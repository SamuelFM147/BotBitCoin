import { useEffect, useState } from "react";

export type Candle = {
  time: number; // ms timestamp
  open: number;
  high: number;
  low: number;
  close: number;
};

export async function fetchCandlesBinance(
  symbol: string = "BTCUSDT",
  interval: string = "1m",
  limit: number = 1440
): Promise<Candle[]> {
  const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Binance error ${res.status}`);
  const data = (await res.json()) as any[];
  return data.map((k) => ({
    time: k[0],
    open: Number(k[1]),
    high: Number(k[2]),
    low: Number(k[3]),
    close: Number(k[4]),
  }));
}

export async function fetchCandlesCoinGecko(days: number = 1): Promise<Candle[]> {
  const url = `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=${days}&interval=minute`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`CoinGecko error ${res.status}`);
  const data = await res.json();
  const prices: [number, number][] = data.prices || [];
  // CoinGecko does not provide OHLC per minute; approximate candles as flat bodies
  return prices.map(([t, p]) => ({ time: t, open: p, high: p, low: p, close: p }));
}

export function usePriceCandles(options?: {
  symbol?: string;
  interval?: string;
  limit?: number;
  refreshMs?: number;
}) {
  const { symbol = "BTCUSDT", interval = "1m", limit = 720, refreshMs = 30000 } = options || {};
  const [data, setData] = useState<Candle[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const candles = await fetchCandlesBinance(symbol, interval, limit);
      setData(candles);
    } catch (e1: any) {
      try {
        const candles = await fetchCandlesCoinGecko(1);
        setData(candles);
      } catch (e2: any) {
        setError(e1?.message || e2?.message || "Failed to fetch price candles");
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, refreshMs);
    return () => clearInterval(id);
  }, [symbol, interval, limit, refreshMs]);

  return { data, error, isLoading: loading, refetch: load };
}