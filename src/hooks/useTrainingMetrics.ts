import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export const useTrainingMetrics = (agentId?: string) => {
  return useQuery({
    queryKey: ["training-metrics", agentId],
    queryFn: async () => {
      try {
        let query = supabase
          .from("training_episodes")
          .select("*")
          .order("episode_number", { ascending: false })
          .limit(100);

        if (agentId) {
          query = query.eq("agent_id", agentId);
        }

        const { data, error } = await query;
        if (!error && data && data.length > 0) {
          return data;
        }
      } catch (_) {
        // Ignora erro e tenta fallback local
      }

      // Fallback local: /public/training_history.json
      const res = await fetch("/training_history.json", { cache: "no-store" });
      if (!res.ok) {
        // Se não houver arquivo local, retorna array vazio para manter o front estável
        return [] as any[];
      }
      const history = await res.json();
      const episodes: number[] = history?.episodes || [];
      const rewards: number[] = history?.rewards || [];
      const losses: number[] = history?.losses || [];

      const mapped = episodes.map((ep: number, idx: number) => ({
        episode_number: ep + 1,
        total_reward: rewards[idx] ?? 0,
        avg_loss: losses[idx] ?? 0,
        agent_id: agentId || history?.agent_id || "local",
      }))
      // Ordena por episódio decrescente para compatibilidade com componentes
      .sort((a: any, b: any) => b.episode_number - a.episode_number);

      return mapped as any[];
    },
    refetchInterval: 2000, // Atualiza a cada 2s para refletir o treino
  });
};
