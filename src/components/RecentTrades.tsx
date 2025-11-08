import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { useRecentTrades } from "@/hooks/useRecentTrades";
import { useMemo } from "react";

interface Trade {
  id: string;
  type: "buy" | "sell";
  price: string;
  amount: string;
  profit: string;
  time: string;
  isProfit: boolean;
}

export const RecentTrades = () => {
  const { data: tradesData, isLoading } = useRecentTrades(5);

  const trades: Trade[] = useMemo(() => {
    if (!tradesData?.length) return [];

    return tradesData.map(trade => {
      const profitLoss = Number(trade.profit_loss || 0);
      const hasProfit = trade.profit_loss !== null;
      
      return {
        id: trade.id,
        type: trade.trade_type as "buy" | "sell",
        price: `$${Number(trade.price).toFixed(2)}`,
        amount: `${Number(trade.amount).toFixed(4)} BTC`,
        profit: hasProfit ? `${profitLoss >= 0 ? '+' : ''}$${Math.abs(profitLoss).toFixed(2)}` : "-",
        time: new Date(trade.timestamp).toLocaleString('pt-BR', { 
          hour: '2-digit', 
          minute: '2-digit',
          day: '2-digit',
          month: '2-digit'
        }),
        isProfit: profitLoss >= 0,
      };
    });
  }, [tradesData]);

  if (isLoading) {
    return (
      <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
        <div className="mb-6">
          <h3 className="text-lg font-semibold">Operações Recentes</h3>
          <p className="text-sm text-muted-foreground">Carregando trades...</p>
        </div>
      </Card>
    );
  }
  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="mb-6">
        <h3 className="text-lg font-semibold">Operações Recentes</h3>
        <p className="text-sm text-muted-foreground">Execuções do agente RL</p>
      </div>
      <div className="space-y-3">
        {trades.map((trade) => (
          <div 
            key={trade.id} 
            className="flex items-center justify-between p-4 rounded-lg bg-secondary/20 hover:bg-secondary/30 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className={`p-2 rounded-lg ${trade.type === 'buy' ? 'bg-success/10' : 'bg-destructive/10'}`}>
                {trade.type === 'buy' ? (
                  <ArrowUpRight className="w-5 h-5 text-success" />
                ) : (
                  <ArrowDownRight className="w-5 h-5 text-destructive" />
                )}
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant={trade.type === 'buy' ? 'default' : 'destructive'} className="text-xs">
                    {trade.type === 'buy' ? 'COMPRA' : 'VENDA'}
                  </Badge>
                  <span className="text-sm font-medium">{trade.price}</span>
                </div>
                <p className="text-sm text-muted-foreground">{trade.amount}</p>
              </div>
            </div>
            <div className="text-right">
              {trade.profit !== "-" && (
                <p className={`text-sm font-semibold ${trade.isProfit ? 'text-success' : 'text-destructive'}`}>
                  {trade.profit}
                </p>
              )}
              <p className="text-xs text-muted-foreground">{trade.time}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};
