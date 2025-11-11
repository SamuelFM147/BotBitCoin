import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchCandlesBinance, fetchCandlesCoinGecko } from "../usePriceCandles";

function mockFetchOnce(response: any, ok = true, status = 200) {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok,
    status,
    json: async () => response,
  })) as any);
}

describe("usePriceCandles helpers", () => {
  afterEach(() => {
    (global as any).fetch && (global as any).fetch.mockRestore?.();
    vi.unstubAllGlobals();
  });

  it("maps Binance klines to Candle[]", async () => {
    const klines = [
      [1731000000000, "100", "110", "95", "105"],
      [1731000060000, "105", "115", "100", "110"],
    ];
    mockFetchOnce(klines);
    const res = await fetchCandlesBinance("BTCUSDT", "1m", 2);
    expect(res.length).toBe(2);
    expect(res[0]).toEqual({ time: 1731000000000, open: 100, high: 110, low: 95, close: 105 });
    expect(res[1].close).toBe(110);
  });

  it("maps CoinGecko prices to flat Candle[]", async () => {
    const payload = { prices: [ [1731000000000, 100], [1731000060000, 110] ] };
    mockFetchOnce(payload);
    const res = await fetchCandlesCoinGecko(1);
    expect(res.length).toBe(2);
    expect(res[0]).toEqual({ time: 1731000000000, open: 100, high: 100, low: 100, close: 100 });
  });
});