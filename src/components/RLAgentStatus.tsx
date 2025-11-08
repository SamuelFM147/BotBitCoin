import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, Target, TrendingUp } from "lucide-react";

export const RLAgentStatus = () => {
  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Brain className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">RL Agent Status</h3>
            <p className="text-sm text-muted-foreground">Modelo: DQN v2.1</p>
          </div>
        </div>
        <Badge className="bg-success/10 text-success border-success/20">
          Ativo
        </Badge>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-primary" />
              <span>Taxa de Exploração</span>
            </div>
            <span className="font-medium">15%</span>
          </div>
          <Progress value={15} className="h-2" />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-primary" />
              <span>Confiança do Modelo</span>
            </div>
            <span className="font-medium">87%</span>
          </div>
          <Progress value={87} className="h-2" />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              <span>Performance (Episódio)</span>
            </div>
            <span className="font-medium">92%</span>
          </div>
          <Progress value={92} className="h-2" />
        </div>

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/50">
          <div>
            <p className="text-sm text-muted-foreground">Episódios</p>
            <p className="text-2xl font-bold">1,247</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Recompensa Média</p>
            <p className="text-2xl font-bold text-success">+$342</p>
          </div>
        </div>
      </div>
    </Card>
  );
};
