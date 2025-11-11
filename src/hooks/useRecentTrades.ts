import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { isSupabaseEnabled } from "@/integrations/supabase/enabled";

export const useRecentTrades = (limit = 10) => {
  return useQuery({
    queryKey: ["recent-trades", limit],
    queryFn: async () => {
      if (isSupabaseEnabled()) {
        try {
          const { data, error } = await supabase
            .from("trades")
            .select("*")
            .order("timestamp", { ascending: false })
            .limit(limit);
          if (!error && data && data.length > 0) {
            return data;
          }
        } catch (_) {
          // Ignora erro e tenta fallback
        }
      }

      // Fallback local: /public/trades.json
      const res = await fetch("/trades.json", { cache: "no-store" });
      if (!res.ok) {
        return [] as any[];
      }
      const trades = await res.json();
      // Normaliza e limita
      const normalized = (Array.isArray(trades) ? trades : [])
        .map((t: any, i: number) => ({
          id: t.id || `local-${i}`,
          trade_type: t.trade_type || "hold",
          price: t.price ?? 0,
          amount: t.amount ?? 0,
          profit_loss: t.profit_loss ?? null,
          timestamp: t.timestamp || new Date().toISOString(),
        }))
        .sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(0, limit);
      return normalized as any[];
    },
    refetchInterval: 2000,
  });
};
