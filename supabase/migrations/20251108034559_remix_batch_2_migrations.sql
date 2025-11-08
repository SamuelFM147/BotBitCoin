
-- Migration: 20251107193546
-- Tabela para armazenar configurações de agentes RL
CREATE TABLE IF NOT EXISTS public.rl_agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  algorithm TEXT NOT NULL CHECK (algorithm IN ('DQN', 'PPO', 'A2C')),
  config JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tabela para episódios de treinamento
CREATE TABLE IF NOT EXISTS public.training_episodes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id UUID NOT NULL REFERENCES public.rl_agents(id) ON DELETE CASCADE,
  episode_number INTEGER NOT NULL,
  total_reward DECIMAL(15, 2) NOT NULL,
  avg_loss DECIMAL(10, 6),
  epsilon DECIMAL(5, 4),
  actions_taken INTEGER,
  duration_seconds INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tabela para trades executados
CREATE TABLE IF NOT EXISTS public.trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id UUID NOT NULL REFERENCES public.rl_agents(id) ON DELETE CASCADE,
  episode_id UUID REFERENCES public.training_episodes(id) ON DELETE SET NULL,
  trade_type TEXT NOT NULL CHECK (trade_type IN ('buy', 'sell', 'hold')),
  price DECIMAL(15, 2) NOT NULL,
  amount DECIMAL(18, 8) NOT NULL,
  profit_loss DECIMAL(15, 2),
  confidence DECIMAL(5, 4),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tabela para métricas de backtesting
CREATE TABLE IF NOT EXISTS public.backtest_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id UUID NOT NULL REFERENCES public.rl_agents(id) ON DELETE CASCADE,
  start_date TIMESTAMPTZ NOT NULL,
  end_date TIMESTAMPTZ NOT NULL,
  total_return DECIMAL(15, 2) NOT NULL,
  sharpe_ratio DECIMAL(10, 4),
  max_drawdown DECIMAL(10, 4),
  win_rate DECIMAL(5, 4),
  total_trades INTEGER,
  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Tabela para dados de mercado históricos
CREATE TABLE IF NOT EXISTS public.market_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  timestamp TIMESTAMPTZ NOT NULL UNIQUE,
  open DECIMAL(15, 2) NOT NULL,
  high DECIMAL(15, 2) NOT NULL,
  low DECIMAL(15, 2) NOT NULL,
  close DECIMAL(15, 2) NOT NULL,
  volume DECIMAL(20, 8) NOT NULL,
  indicators JSONB DEFAULT '{}'::jsonb
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_training_episodes_agent ON public.training_episodes(agent_id, episode_number DESC);
CREATE INDEX IF NOT EXISTS idx_trades_agent ON public.trades(agent_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_episode ON public.trades(episode_id);
CREATE INDEX IF NOT EXISTS idx_backtest_agent ON public.backtest_results(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON public.market_data(timestamp DESC);

-- RLS Policies (dados públicos para demonstração)
ALTER TABLE public.rl_agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.backtest_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_data ENABLE ROW LEVEL SECURITY;

-- Permitir leitura pública para visualização (ajustar conforme necessário)
CREATE POLICY "Allow public read on rl_agents" ON public.rl_agents FOR SELECT USING (true);
CREATE POLICY "Allow public read on training_episodes" ON public.training_episodes FOR SELECT USING (true);
CREATE POLICY "Allow public read on trades" ON public.trades FOR SELECT USING (true);
CREATE POLICY "Allow public read on backtest_results" ON public.backtest_results FOR SELECT USING (true);
CREATE POLICY "Allow public read on market_data" ON public.market_data FOR SELECT USING (true);

-- Trigger para updated_at
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rl_agents_updated_at
  BEFORE UPDATE ON public.rl_agents
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Migration: 20251107200217
-- Corrigir função para incluir search_path
DROP FUNCTION IF EXISTS public.update_updated_at_column() CASCADE;

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- Recriar o trigger
CREATE TRIGGER update_rl_agents_updated_at
  BEFORE UPDATE ON public.rl_agents
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();
