import { TradingMetrics } from "@/components/TradingMetrics";
import { PriceChart } from "@/components/PriceChart";
import { RLAgentStatus } from "@/components/RLAgentStatus";
import { TrainingProgress } from "@/components/TrainingProgress";
import { RecentTrades } from "@/components/RecentTrades";
import { RiskManagement } from "@/components/RiskManagement";
import { Bot } from "lucide-react";
import { useTrainingStatus } from "@/hooks/useTrainingStatus";

const Index = () => {
  const { data: status } = useTrainingStatus();
  const lastUpdatedLabel = (() => {
    if (!status?.timestamp) return "Sem dados";
    const ts = new Date(status.timestamp).getTime();
    const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (diffSec < 60) return `Atualizado há ${diffSec}s`;
    const mins = Math.floor(diffSec / 60);
    return `Atualizado há ${mins} min`;
  })();
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-primary to-primary/70">
                <Bot className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Bitcoin RL Trading System</h1>
                <p className="text-sm text-muted-foreground">Sistema de Trading com Reinforcement Learning</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Sistema v2.1.0</p>
              <p className="text-xs text-muted-foreground">{lastUpdatedLabel}</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Metrics Overview */}
          <section>
            <TradingMetrics />
          </section>

          {/* Charts Section */}
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PriceChart />
            </div>
            <div>
              <RLAgentStatus />
            </div>
          </section>

          {/* Training and Trades */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TrainingProgress />
            <RecentTrades />
          </section>

          {/* Risk Management */}
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <RiskManagement />
            </div>
            <div className="lg:col-span-2">
              {/* Placeholder for future components */}
              <div className="h-full rounded-lg border border-dashed border-border/50 flex items-center justify-center p-6">
                <div className="text-center">
                  <p className="text-muted-foreground mb-2">Área reservada para análises adicionais</p>
                  <p className="text-sm text-muted-foreground">Backtesting detalhado, feature importance, etc.</p>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 bg-card/30 backdrop-blur-sm mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center text-sm text-muted-foreground">
            <p>Sistema de Trading Automatizado com RL • DQN Architecture • Bitcoin Market</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
