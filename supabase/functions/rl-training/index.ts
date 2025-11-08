import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const { action, data } = await req.json();

    switch (action) {
      case 'save_episode':
        const { data: episode, error: episodeError } = await supabase
          .from('training_episodes')
          .insert({
            agent_id: data.agent_id,
            episode_number: data.episode_number,
            total_reward: data.total_reward,
            avg_loss: data.avg_loss,
            epsilon: data.epsilon,
            actions_taken: data.actions_taken,
            duration_seconds: data.duration_seconds
          })
          .select()
          .single();

        if (episodeError) throw episodeError;
        return new Response(JSON.stringify({ success: true, episode }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      case 'save_trade':
        const { data: trade, error: tradeError } = await supabase
          .from('trades')
          .insert({
            agent_id: data.agent_id,
            episode_id: data.episode_id,
            trade_type: data.trade_type,
            price: data.price,
            amount: data.amount,
            profit_loss: data.profit_loss,
            confidence: data.confidence
          })
          .select()
          .single();

        if (tradeError) throw tradeError;
        return new Response(JSON.stringify({ success: true, trade }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      case 'get_metrics':
        const { data: metrics, error: metricsError } = await supabase
          .from('training_episodes')
          .select('*')
          .eq('agent_id', data.agent_id)
          .order('episode_number', { ascending: false })
          .limit(100);

        if (metricsError) throw metricsError;
        return new Response(JSON.stringify({ success: true, metrics }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      default:
        return new Response(JSON.stringify({ error: 'Invalid action' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
    }
  } catch (error) {
    console.error('Error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});
