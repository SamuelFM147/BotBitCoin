import { Card } from "@/components/ui/card";
import { useRef, useMemo, useEffect, useState } from "react";
import { usePriceCandles } from "@/hooks/usePriceCandles";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import {
  VictoryAxis,
  VictoryCandlestick,
  VictoryChart,
  VictoryTheme,
  VictoryZoomContainer,
} from "victory";

export const PriceChart = () => {
  const { data: candles, error, isLoading } = usePriceCandles({ symbol: "BTCUSDT", interval: "1m", limit: 720 });
  const latest = useMemo(() => (candles && candles.length ? candles[candles.length - 1] : null), [candles]);
  const latestPrice = latest?.close ?? 0;
  const changePct = useMemo(() => {
    if (!candles || candles.length < 2) return 0;
    const first = candles[0].close;
    const last = candles[candles.length - 1].close;
    return ((last - first) / first) * 100;
  }, [candles]);

  const [containerWidth, setContainerWidth] = useState<number>(800);
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width ?? 800;
      setContainerWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const victoryData = useMemo(
    () => (candles || []).map((c) => ({ x: new Date(c.time), open: c.open, close: c.close, high: c.high, low: c.low })),
    [candles]
  );

  const fallbackData = useMemo(
    () => (candles || []).map((c) => ({ time: new Date(c.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), price: c.close })),
    [candles]
  );

  const priceColor = changePct >= 0 ? "text-success" : "text-destructive";
  const priceSign = changePct >= 0 ? "+" : "";

  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">Bitcoin Price</h3>
          <p className="text-sm text-muted-foreground">Últimas 24 horas</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold">{latestPrice ? `$${latestPrice.toLocaleString()}` : "—"}</p>
          <p className={`text-sm ${priceColor} flex items-center gap-1 justify-end`}>
            {`${priceSign}${changePct.toFixed(2)}%`}
          </p>
        </div>
      </div>

      {/* Candles (Victory) with fallback to Area (Recharts) */}
      <div ref={containerRef} style={{ width: "100%", height: 300 }}>
        {error && !isLoading && fallbackData.length === 0 ? (
          <div className="text-sm text-muted-foreground">Falha ao carregar dados de preço: {error}</div>
        ) : candles && candles.length ? (
          <VictoryChart
            theme={VictoryTheme.material}
            domainPadding={{ x: 15, y: 15 }}
            width={containerWidth}
            height={300}
            scale={{ x: "time" }}
            containerComponent={<VictoryZoomContainer zoomDimension="x" />}
          >
            <VictoryAxis tickFormat={(t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} />
            <VictoryAxis dependentAxis tickFormat={(y) => `$${y.toFixed(0)}`}/> 
            <VictoryCandlestick
              candleColors={{ positive: "#16a34a", negative: "#ef4444" }}
              data={victoryData}
            />
          </VictoryChart>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={fallbackData}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={["dataMin - 500", "dataMax + 500"]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <Area type="monotone" dataKey="price" stroke="hsl(var(--primary))" strokeWidth={2} fill="url(#colorPrice)" />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
};
