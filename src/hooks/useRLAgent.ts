import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export const useRLAgent = (agentId?: string) => {
  return useQuery({
    queryKey: ["rl-agent", agentId],
    queryFn: async () => {
      if (agentId) {
        const { data, error } = await supabase
          .from("rl_agents")
          .select("*")
          .eq("id", agentId)
          .single();
        
        if (error) throw error;
        return data;
      } else {
        const { data, error } = await supabase
          .from("rl_agents")
          .select("*")
          .order("created_at", { ascending: false })
          .limit(1)
          .single();
        
        if (error) throw error;
        return data;
      }
    },
  });
};
