import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useTrainingMetrics } from "@/hooks/useTrainingMetrics";
import { useMemo } from "react";

export const TrainingProgress = () => {
  const { data: episodes, isLoading } = useTrainingMetrics();

  const trainingData = useMemo(() => {
    if (!episodes?.length) {
      return [
        { episode: 0, reward: 0, loss: 0 },
      ];
    }

    return episodes
      .slice()
      .reverse()
      .map(ep => ({
        episode: ep.episode_number,
        reward: Number(ep.total_reward),
        loss: Number(ep.avg_loss || 0),
      }));
  }, [episodes]);

  if (isLoading) {
    return (
      <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
        <div className="mb-6">
          <h3 className="text-lg font-semibold">Progresso de Treinamento</h3>
          <p className="text-sm text-muted-foreground">Carregando dados...</p>
        </div>
      </Card>
    );
  }
  return (
    <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="mb-6">
        <h3 className="text-lg font-semibold">Progresso de Treinamento</h3>
        <p className="text-sm text-muted-foreground">Evolução de recompensa por episódio</p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={trainingData}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
          <XAxis 
            dataKey="episode" 
            stroke="hsl(var(--muted-foreground))" 
            fontSize={12}
            label={{ value: 'Episódios', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            stroke="hsl(var(--muted-foreground))" 
            fontSize={12}
            label={{ value: 'Recompensa ($)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="reward" 
            stroke="hsl(var(--chart-2))" 
            strokeWidth={2}
            name="Recompensa"
            dot={{ fill: 'hsl(var(--chart-2))' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
};
