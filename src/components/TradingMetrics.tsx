import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Activity, DollarSign } from "lucide-react";
import { useTrainingMetrics } from "@/hooks/useTrainingMetrics";
import { useRecentTrades } from "@/hooks/useRecentTrades";
import { useMemo } from "react";

interface MetricCardProps {
  title: string;
  value: string;
  change: string;
  isPositive: boolean;
  icon: React.ReactNode;
}

const MetricCard = ({ title, value, change, isPositive, icon }: MetricCardProps) => (
  <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all duration-300">
    <div className="flex items-center justify-between mb-4">
      <div className="p-2 rounded-lg bg-primary/10">
        {icon}
      </div>
      <div className={`flex items-center gap-1 text-sm font-medium ${isPositive ? 'text-success' : 'text-destructive'}`}>
        {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
        {change}
      </div>
    </div>
    <div>
      <p className="text-sm text-muted-foreground mb-1">{title}</p>
      <p className="text-3xl font-bold">{value}</p>
    </div>
  </Card>
);

export const TradingMetrics = () => {
  const { data: episodes } = useTrainingMetrics();
  const { data: trades } = useRecentTrades(100);

  const metrics = useMemo(() => {
    if (!episodes?.length || !trades?.length) {
      return [
        {
          title: "Total de Retorno",
          value: "$0",
          change: "0%",
          isPositive: true,
          icon: <DollarSign className="w-5 h-5 text-primary" />,
        },
        {
          title: "Win Rate",
          value: "0%",
          change: "0%",
          isPositive: true,
          icon: <Activity className="w-5 h-5 text-primary" />,
        },
        {
          title: "Episódios",
          value: "0",
          change: "0",
          isPositive: true,
          icon: <TrendingUp className="w-5 h-5 text-primary" />,
        },
        {
          title: "Trades Totais",
          value: "0",
          change: "0",
          isPositive: true,
          icon: <Activity className="w-5 h-5 text-primary" />,
        },
      ];
    }

    const totalReward = episodes.reduce((sum, ep) => sum + Number(ep.total_reward), 0);
    const latestEpisode = episodes[0];
    const previousEpisode = episodes[1];
    
    const winningTrades = trades.filter(t => Number(t.profit_loss) > 0).length;
    const winRate = (winningTrades / trades.length) * 100;
    
    const totalProfitLoss = trades.reduce((sum, t) => sum + Number(t.profit_loss || 0), 0);
    
    const rewardChange = previousEpisode 
      ? ((Number(latestEpisode.total_reward) - Number(previousEpisode.total_reward)) / Math.abs(Number(previousEpisode.total_reward)) * 100)
      : 0;

    return [
      {
        title: "Total de Retorno",
        value: `$${totalProfitLoss.toFixed(2)}`,
        change: `${rewardChange >= 0 ? '+' : ''}${rewardChange.toFixed(1)}%`,
        isPositive: totalProfitLoss >= 0,
        icon: <DollarSign className="w-5 h-5 text-primary" />,
      },
      {
        title: "Win Rate",
        value: `${winRate.toFixed(1)}%`,
        change: `${winningTrades} vitórias`,
        isPositive: winRate >= 50,
        icon: <Activity className="w-5 h-5 text-primary" />,
      },
      {
        title: "Episódios",
        value: episodes.length.toString(),
        change: `Ep ${latestEpisode.episode_number}`,
        isPositive: true,
        icon: <TrendingUp className="w-5 h-5 text-primary" />,
      },
      {
        title: "Trades Totais",
        value: trades.length.toString(),
        change: `${trades.filter(t => t.trade_type === 'buy').length} compras`,
        isPositive: true,
        icon: <Activity className="w-5 h-5 text-primary" />,
      },
    ];
  }, [episodes, trades]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, index) => (
        <MetricCard key={index} {...metric} />
      ))}
    </div>
  );
};
