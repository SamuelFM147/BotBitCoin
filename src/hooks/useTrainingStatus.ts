import { useQuery } from "@tanstack/react-query";

export interface TrainingStatus {
  episode_number: number;
  epsilon: number;
  device: string;
  gpu_available: boolean;
  reward: number;
  loss: number;
  steps: number;
  portfolio_value: number;
  profit: number;
  n_trades: number;
  timestamp: string;
  running: boolean;
  agent_id?: string;
}

export const useTrainingStatus = () => {
  return useQuery<TrainingStatus | null>({
    queryKey: ["training-status"],
    queryFn: async () => {
      const res = await fetch("/training_status.json", { cache: "no-store" });
      if (!res.ok) return null;
      const data = await res.json();
      return data as TrainingStatus;
    },
    refetchInterval: 2000,
  });
};