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
    const contentType = req.headers.get('content-type') || '';
    if (!contentType.toLowerCase().includes('application/json')) {
      return new Response(JSON.stringify({ error: 'Content-Type deve ser application/json' }), {
        status: 415,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    const reqHost = new URL(req.url).host;
    const derivedUrl = `https://${reqHost.replace('.functions.supabase.co', '.supabase.co')}`;
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? derivedUrl;
    const authHeader = req.headers.get('authorization') || '';
    const headerToken = authHeader.replace(/^Bearer\s+/i, '').trim();
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? headerToken ?? (Deno.env.get('SUPABASE_ANON_KEY') ?? '');

    const supabase = createClient(supabaseUrl, supabaseKey);

    const { action, data } = await req.json();

    // Helper: resolve or create agent by name, returning UUID id
    const ensureAgentId = async (identifier: string): Promise<string> => {
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
      // If a UUID was provided and exists, use it directly
      if (uuidRegex.test(identifier)) {
        const { data: byId, error: byIdError } = await supabase
          .from('rl_agents')
          .select('id')
          .eq('id', identifier)
          .limit(1);
        if (byIdError) throw byIdError;
        if (byId && byId.length > 0) return identifier;
      }

      // Try to find by name
      const { data: byName, error: byNameError } = await supabase
        .from('rl_agents')
        .select('id')
        .eq('name', identifier)
        .limit(1);
      if (byNameError) throw byNameError;
      if (byName && byName.length > 0) return byName[0].id as string;

      // Create if not found
      const { data: inserted, error: insertError } = await supabase
        .from('rl_agents')
        .insert({ name: identifier, algorithm: 'DQN', config: {} })
        .select('id')
        .limit(1);
      if (insertError) throw insertError;
      if (!inserted || inserted.length === 0) throw new Error('Failed to create rl_agent');
      return inserted[0].id as string;
    };

    // Helper: normalize trade type to schema-allowed values
    const normalizeTradeType = (t: string): 'buy' | 'sell' | 'hold' => {
      const v = String(t || '').toLowerCase();
      if (v === 'buy' || v === 'sell' || v === 'hold') return v as 'buy' | 'sell' | 'hold';
      if (v === 'auto_close' || v === 'close' || v === 'exit') return 'sell';
      return 'hold';
    };

    switch (action) {
      case 'save_episode':
        // Ensure agent exists and get UUID
        const agentId = await ensureAgentId(String(data.agent_id));
        const { data: episode, error: episodeError } = await supabase
          .from('training_episodes')
          .insert({
            agent_id: agentId,
            episode_number: Number(data.episode_number ?? 0),
            total_reward: Number(data.total_reward ?? 0),
            avg_loss: data.avg_loss !== undefined ? Number(data.avg_loss) : null,
            epsilon: data.epsilon !== undefined ? Number(data.epsilon) : null,
            actions_taken: data.actions_taken !== undefined ? Number(data.actions_taken) : null,
            duration_seconds: data.duration_seconds !== undefined ? Math.round(Number(data.duration_seconds)) : null,
          })
          .select()
          .single();

        if (episodeError) throw episodeError;
        return new Response(JSON.stringify({ success: true, episode }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      case 'save_trade':
        // Ensure agent exists and get UUID
        const agentIdForTrade = await ensureAgentId(String(data.agent_id));
        const normalizedType = normalizeTradeType(data.trade_type);
        const { data: trade, error: tradeError } = await supabase
          .from('trades')
          .insert({
            agent_id: agentIdForTrade,
            episode_id: String(data.episode_id || ''),
            trade_type: normalizedType,
            price: Number(data.price ?? 0),
            amount: Number(data.amount ?? 0),
            profit_loss: data.profit_loss !== undefined ? Number(data.profit_loss) : null,
            confidence: data.confidence !== undefined ? Number(data.confidence) : null,
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
    const safeMessage = (() => {
      try {
        if (error && typeof error === 'object') {
          const anyErr = error as any;
          return anyErr.message || anyErr.error_description || anyErr.error || JSON.stringify(anyErr);
        }
        return String(error);
      } catch (_) {
        return 'Erro desconhecido';
      }
    })();
    console.error('rl-training error:', safeMessage, error);
    return new Response(JSON.stringify({ error: safeMessage }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});
