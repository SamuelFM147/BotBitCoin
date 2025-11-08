import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Shield, AlertTriangle, CheckCircle2, Activity } from "lucide-react";

export const RiskManagement = () => {
  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-lg bg-primary/10">
          <Shield className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Gerenciamento de Risco</h3>
          <p className="text-sm text-muted-foreground">Indicadores de proteção</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-success" />
              <span className="text-sm">Exposição Atual</span>
            </div>
            <span className="text-sm font-medium">42%</span>
          </div>
          <Progress value={42} className="h-2" />
          <p className="text-xs text-muted-foreground">Limite: 60% do capital</p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-warning" />
              <span className="text-sm">Volatilidade</span>
            </div>
            <span className="text-sm font-medium">Média</span>
          </div>
          <Progress value={55} className="h-2" />
          <p className="text-xs text-muted-foreground">VIX Bitcoin: 55</p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-destructive" />
              <span className="text-sm">Stop Loss Ativo</span>
            </div>
            <span className="text-sm font-medium">5 posições</span>
          </div>
          <Progress value={35} className="h-2" />
          <p className="text-xs text-muted-foreground">Nível médio: -3.2%</p>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/50">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Capital Disponível</p>
            <p className="text-lg font-bold">$72,400</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Em Posições</p>
            <p className="text-lg font-bold">$52,180</p>
          </div>
        </div>
      </div>
    </Card>
  );
};
