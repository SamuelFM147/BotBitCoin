import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export const useRecentTrades = (limit = 10) => {
  return useQuery({
    queryKey: ["recent-trades", limit],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("trades")
        .select("*")
        .order("timestamp", { ascending: false })
        .limit(limit);
      
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
};
