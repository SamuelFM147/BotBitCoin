import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, Target, TrendingUp } from "lucide-react";
import { useTrainingStatus } from "@/hooks/useTrainingStatus";

export const RLAgentStatus = () => {
  const { data: status } = useTrainingStatus();
  const epsilonPct = Math.round(((status?.epsilon ?? 0) * 100 + Number.EPSILON) * 100) / 100;
  const running = status?.running ?? false;
  const device = status?.device ?? "cpu";
  const episode = status?.episode_number ?? 0;
  const gpuAvailable = status?.gpu_available ?? false;
  const trades = status?.n_trades ?? 0;
  const reward = status?.reward ?? 0;
  const loss = status?.loss ?? 0;

  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Brain className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">RL Agent Status</h3>
            <p className="text-sm text-muted-foreground">Modelo: DQN v2.1 • Episódio {episode}</p>
          </div>
        </div>
        <Badge className={running ? "bg-success/10 text-success border-success/20" : "bg-destructive/10 text-destructive border-destructive/20"}>
          {running ? "Ativo" : "Parado"}
        </Badge>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-primary" />
              <span>Taxa de Exploração</span>
            </div>
            <span className="font-medium">{epsilonPct}%</span>
          </div>
          <Progress value={epsilonPct} className="h-2" />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-primary" />
              <span>Dispositivo</span>
            </div>
            <span className="font-medium">{device} {gpuAvailable ? "(GPU)" : "(CPU)"}</span>
          </div>
          <Progress value={gpuAvailable ? 100 : 50} className="h-2" />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              <span>Recompensa / Loss (episódio)</span>
            </div>
            <span className="font-medium">{reward.toFixed(2)} / {loss.toFixed(4)}</span>
          </div>
          <Progress value={Math.max(0, Math.min(100, (reward !== 0 ? 50 + Math.sign(reward) * 50 : 50)))} className="h-2" />
        </div>

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/50">
          <div>
            <p className="text-sm text-muted-foreground">Episódios</p>
            <p className="text-2xl font-bold">{episode}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Recompensa Média</p>
            <p className="text-2xl font-bold text-success">{reward >= 0 ? "+" : "-"}${Math.abs(reward).toFixed(2)}</p>
          </div>
        </div>
      </div>
    </Card>
  );
};
