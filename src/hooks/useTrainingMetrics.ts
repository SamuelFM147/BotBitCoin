import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export const useTrainingMetrics = (agentId?: string) => {
  return useQuery({
    queryKey: ["training-metrics", agentId],
    queryFn: async () => {
      let query = supabase
        .from("training_episodes")
        .select("*")
        .order("episode_number", { ascending: false })
        .limit(100);

      if (agentId) {
        query = query.eq("agent_id", agentId);
      }

      const { data, error } = await query;
      
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000, // Refetch every 5 seconds for real-time updates
  });
};
