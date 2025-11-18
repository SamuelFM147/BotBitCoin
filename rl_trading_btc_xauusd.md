# Aprendizado por Reforço (RL) aplicado a Trading — foco em BTC e XAUUSD

> **Resumo:** Este documento explica como algoritmos de Aprendizado por Reforço (RL) são formulados e aplicados a trading, e propõe um *pipeline* prático — do _data engineering_ ao deploy — para construir um modelo de IA de trading para **BTC** e **XAUUSD**. Ao final há notas específicas por ativo e pitfalls comuns.

---

## 1) Como o RL é formulado para trading

**MDP/POMDP.** No RL clássico, um agente interage com um ambiente em etapas: observa um **estado** \(S_t\), escolhe uma **ação** \(A_t\) via política \(\pi(A|S)\), recebe uma **recompensa** \(R_{t+1}\) e transita para \(S_{t+1}\). Em trading, o ambiente é o mercado; como não observamos tudo (ordens de terceiros, inventário, latência etc.), lidamos de fato com um **POMDP**: trabalhamos com uma **observação** \(X_t = O(S_t)\) construída a partir de preços, volumes, livro de ofertas, posição atual, caixa, ordens abertas etc. (conceitualização útil em Britz, 2018).

**Estado/observação.** Exemplos práticos:
- Janelas temporais de OHLCV (1m–1h), retornos, volatilidade realizada, _features_ técnicas.
- Sinais de microestrutura: _order flow imbalance_, profundidade do book, _queue lengths_.
- Estado da conta: posição, caixa, margem, PnL não realizado, ordens pendentes.

**Espaço de ações.**
- **Discreto:** {Comprar, Vender, Manter} ou {+1, 0, −1} exposição.
- **Contínuo:** tamanho de posição (fração do capital), *leverage*, preço/quantidade de **ordens limite** (preço e quantidade contínuos), cancelamentos — essencial para execução mais realista.
- Ambientes de futuros/FX costumam funcionar bem com espaços contínuos quando queremos dimensionar posição e controlar risco diretamente.

**Função de recompensa.** Opções frequentes:
- **PnL realizado** por passo, líquido de custos.
- **Retorno ajustado à volatilidade** (ex.: retorno / volatilidade alvo) para estabilizar aprendizado.
- **Reward _risk-aware_**: termos de penalização para *drawdown*, _variance_, custo de transação, *inventory risk* e violação de limites. A literatura mostra bons resultados ao **escalar posições pela volatilidade** no próprio _reward_ para que a política se adapte a regimes de mercado.

**Por que RL (vs. previsão + regras)?** Em vez de prever e depois mapear previsões em decisões, o RL **aprende diretamente a política de posição/execução** e pode internalizar fricções (corretagem, *slippage*) no objetivo de otimização, evitando um _gap_ entre sinal e execução.

---

## 2) Evidências/publicações úteis

- **Futuros (multi‑ativos):** autores da Univ. de Oxford mostram RL discreto/contínuo com *volatility scaling* e resultados positivos (mesmo com custos) em 50 contratos de 2011–2019, cobrindo **commodities, FX e índices**.  
- **Bitcoin (BTC):** uso combinado de **Dueling Double DQN**, **PPO** e **A2C**, com avaliação por **Sharpe** e lucro, reportando melhor desempenho quando integram múltiplos agentes.  
- **XAUUSD (ouro spot/CFD):** estudo recente combina **deep learning no livro de ofertas** com **temporal‑difference RL** e testa 5 instrumentos, incluindo **XAUUSD**, com *backtests* e *forward tests* em plataforma varejo; mostra potencial, mas ressalta lidar com custos, *slippage* e spread.

> Referências completas no final.

---

## 3) Pipeline recomendado (de ponta a ponta)

Abaixo um _blueprint_ enxuto e reproduzível, do zero ao _paper trading_.

### 3.1 Dados & engenharia
1. **Fontes:**  
   - **BTC:** exchange(s) com histórico de trades/quotes e fee schedule; se usar perp/futuros, incluir funding/financiamento.  
   - **XAUUSD:** corretoras/fornecedores de FX/CFD com _ticks_ e **livro de ofertas**; horários 23×5 e _rolls_/sessões (Londres/NY).
2. **Granularidade:** 1m–5m para intraday “inteligente”, ou *event‑driven* por mudanças no livro de ofertas.  
3. **Limpeza:** sincronização de fusos, remoção de *bad ticks*, _resampling_, reconstrução de *midprice*, *bid‑ask spread* e **custos transacionais** realistas.  
4. **_Features_ (opcional):** retornos normalizados, volatilidade, RSI/MACD apenas como *hints* (não obrigatório). Em HFT, priorize sinais de **microestrutura** (OFI, profundidade, desequilíbrio).  
5. **Partição temporal:** *train/validation/test* com **cortes por tempo** (sem vazamento). Use **_walk‑forward_**.

### 3.2 Definição do ambiente (Gym‑like)
- **Observação \(X_t\):** pilha de _frames_ (janelas) com OHLCV e/ou *book* (N níveis), posição, caixa, PnL flutuante.  
- **Ações:**  
  - *Discreto:* {long/flat/short}.  
  - *Contínuo:* \(\Delta\) posição \(\in [-\lambda, +\lambda]\), ou ordens limite \((p,q)\) com validade.  
- **Transição:** simular execução (prioridade por preço/tempo), latência artificial (p.ex. 100–300ms), **slippage** estocástico e **comissões**.  
- **Recompensa \(r_t\):**  
  \[ r_t = \frac{\text{PnL}_t - \text{cost}_t}{\sigma_\text{target}} - \lambda_\text{DD}\cdot \Delta \text{DD}_t - \lambda_\text{inv}\cdot|pos_t| \]  
  onde \(\sigma_\text{target}\) fornece **_volatility scaling_**; incluir _maker/taker fees_, *spreads* e impostos conforme o ativo.  
- **Restrições:** limites de alavancagem, margens, risco por trade/dia; *kill‑switch* por *max drawdown* intradiário.

### 3.3 Algoritmos & arquitetura
- **Ações discretas:** DQN/DDQN/DuelingDQN (+ *prioritized replay*).  
- **Contínuas:** **PPO**, **A2C/A3C**, **TD3**, **SAC**; para ordens limite, combinar política de **preço** e **quantidade** (duas cabeças).  
- **_Representation learning_:** 1D‑CNNs para séries; _transformers_ temporais; para **order book**, CNN/ResNet 2D ou redes específicas de microestrutura.  
- **Truques úteis:** *reward clipping*, *volatility targeting*, *action smoothing*, *entropy bonus*, *risk‑aware losses*.

### 3.4 Treinamento
- **Curriculum por regimes:** treinar por blocos de volatilidade/mercado (alta, baixa, lateral).  
- **Randomização de custos/latência:** aumenta robustez a *slippage* e spreads variáveis.  
- **Validação:** _rolling_ **walk‑forward**; *early stopping* por métricas fora da amostra.  
- **Ablations:** sem _features_, sem _vol scaling_, sem custos — para medir contribuição.

### 3.5 Avaliação
- **Métricas:** **Sharpe**, **Sortino**, **Calmar**, **Max Drawdown**, **Hit‑rate**, **Avg trade**, **Turnover**, **Capacity**.  
- **Robustez:** _bootstrapping_ de ordens, **estresse de custos** (+50–200%), estresse de latência, *gap risk*.  
- **Benchmarks:** *buy‑and‑hold* (BTC), *carry* (perp/futuros), **time‑series momentum** e _signals_ simples.

### 3.6 Transição para produção
- **Paper trading** (conta demo) ≥ 4–8 semanas, com monitoramento de *PnL*, desvios vs. backtest e alarmes.  
- **Controle de risco em tempo real:** _volatility targeting_ diário, _hard stops_, circuit‑breakers por **DD**.  
- **MLOps:** _versioning_ (dados/modelo), _feature store_ determinística, *retraining* periódico, **detecção de drift** (KS/adfuller).  
- **Observabilidade:** *latency budgets*, rejeições, *fill ratio*, divergência de *slippage*.

---

## 4) Notas por ativo

**BTC (24×7):**
- Liquidez contínua, *funding* em perp/futuros e **spreads** variáveis por horário/exchange.  
- Estratégias com **tendência** funcionam bem com **vol scaling** e _position holding_ durante _trends_; incluir custos de retirada/deposito e risco de exchange.  

**XAUUSD (23×5):**
- Ciclo diário (Londres/NY) com mudanças de **spread**/liquidez; *rolls* e *fixings* importam.  
- Se usar **livro de ofertas**, _order‑flow_ e **desequilíbrio** são sinais úteis; cuidado com _slippage_ em notícias macro (CPI, NFP, FOMC).  
- Teste **TD‑RL** com *features* de microestrutura (OFI), avaliado via _backtest_ + _forward test_.

---

## 5) Esqueleto do ambiente (pseudocódigo)

```python
class TradingEnv(gym.Env):
    def __init__(self, data, costs, lat_ms=150, action_space="continuous"):
        # buffers de observação (OHLCV/book), posição, caixa, pnl
        ...

    def step(self, action):
        # 1) mapear ação -> ordens/posição (incl. limites, tamanho, cancelações)
        # 2) simular execução com latência, comissões, slippage e spread
        # 3) atualizar posição/pnl/custos e observação
        reward = (pnl_delta - costs_now)/sigma_target - lam_dd*dd_delta - lam_inv*abs(pos)
        done = dd>dd_max or t>=T_end
        return obs, reward, done, info

    def reset(self):
        # reinicializa buffers e estado de risco
        ...
```

---

## 6) Checklist mínimo para ir a mercado

- [ ] Dados limpos, determinísticos e *replayable*.  
- [ ] Ambiente reproduzíveis com custos/latência estocásticos.  
- [ ] **Validação _walk‑forward_** com *stress* de custos.  
- [ ] Paper trading aprovado com **Sharpe > 1** e **DD** dentro do limite.  
- [ ] Monitoramento/alertas de risco e *drift* prontos.  

---

## 7) Referências (selecionadas)

1. **Deep Reinforcement Learning for Trading** — *Journal of Financial Data Science* (Oxford‑Man Institute). Discutem ações discretas/contínuas e **volatility scaling**; resultados positivos mesmo com custos em 50 futuros (2011–2019).  
2. **Deep Reinforcement Learning for Bitcoin Trading** — Integra **Dueling Double DQN**, **PPO** e **A2C**; mede por **Sharpe** e lucro, reportando melhor resultado ao combinar agentes.  
3. **Combining Deep Learning on Order Books with RL for Profitable Trading** — *arXiv 2311.02088*. Usa **TD‑RL** com *order‑flow imbalance*; testa GBPUSD, EURUSD, DE40, FTSE100 e **XAUUSD**; mostra potencial e ressalta custos/slippage/spread.  
4. **Learning to Trade with RL** — Denny Britz (2018). Visão prática: MDP→trading, POMDP, ação discreta vs. contínua, desenho de recompensa.

---

# Roadmap dos Ambientes de Trading (BTC e XAUUSD)

> Objetivo: consolidar um plano executável, faseado e mensurável para construir ambientes de trading realistas (simulação, backtest e produção) com agentes de RL, cobrindo dados, arquitetura de ambiente, recompensas, algoritmos, validação, UI e DevOps. Execução sempre via `poetry run` em Windows.

## Fase 0 — Fundamentos e Governança
- Estrutura: `src/` (código), `tests/` (pytest), `configs/` (YAML), `runs/` (logs), `checkpoints/` (modelos), `.env` (não versionado).
- Padrões: PEP 8, type hints em Python, nomes em inglês e docstrings em português; seeds fixos e rastreamento de experimentos.
- Comandos: `poetry install`, `poetry run pytest -q`, `poetry run ruff check .`, `poetry run mypy .`.
- Gate: suíte mínima verde, qualidade estática sem erros críticos.

## Fase 1 — Dados e Feature Engineering
- Coleta: OHLCV para BTC e XAUUSD em `1m/5m/1h`; se disponível, livro de ofertas (N níveis) e funding (perp). Timezone normalizado.
- Limpeza: tratamento de faltas/outliers, reconstrução de `midprice`, cálculo de `bid-ask spread`, sincronização entre ativos.
- Partição: `train/validation/test` por tempo; walk-forward.
- Features: retornos log, volatilidade (rolling std/ATR), momentum (EMA/SMA), RSI/MACD, microestrutura (OFI, imbalance), normalização z-score/robust.
- Entregáveis: pipelines reproduzíveis e validações; artefatos em cache.
- Gate: cobertura de dados > 95%, ausência de look-ahead.

## Fase 2 — Ambiente Gymnasium (Simulador de Mercado)
- Observações: janela `T×F` de preços/indicadores, posição, caixa, PnL não realizado.
- Ações: Discreto `{-1,0,+1}` (short/flat/long) e/ou Contínuo `[-1,+1]` alvo de posição.
- Execução: custos (`fees_bps`, `slippage_bps`), latência (`latency_steps`), ordens `market/limit`, restrições (`max_leverage`, `position_limit`).
- Recompensa: `dlogNAV - costs - λ_vol*vol - λ_turn*|Δpos| - λ_dd*drawdown` com parâmetros configuráveis.
- Entregáveis: `envs/btc_xau_env.py` compatível com Gymnasium.
- Gate: testes de sanidade de PnL/custos e reprodutibilidade.

## Fase 3 — Backtester e Avaliação
- Backtest event-driven com execução realista (spread, comissões, slippage, latência); logs de ordens e reconciliação.
- Métricas: `CAGR`, `Sharpe`, `Sortino`, `MaxDD`, `Profit Factor`, `Turnover`, `Exposure`.
- Validação: walk-forward, Monte Carlo (bootstrap de blocos), stress de custos/latência.
- Entregáveis: `evaluation/backtester.py` e `evaluation/metrics.py` com testes.
- Gate: métricas estáveis e consistentes em múltiplos regimes.

## Fase 4 — Agentes RL (Baselines e Progressão)
- Discreto: `PPO` e `DQN`/`Double+Dueling` com `PER` e `n-step`.
- Contínuo: `SAC`/`TD3` para alvo de posição.
- Arquiteturas: CNN 1D; LSTM/GRU; Transformer leve (atenção temporal).
- Estratégias: currículo de custos/latência; normalização online; exploração (entropy/epsilon/noisy).
- Entregáveis: `agents/` e `trainers/` com `early_stopping` e checkpoints.
- Gate: superar baselines (buy-and-hold/momentum) out-of-sample.

## Fase 5 — Aprimoramentos e Risco
- Reward `risk-aware` (penalização de volatilidade/drawdown/turnover) e `volatility targeting`.
- Portfolio BTC+XAU com `risk-parity` e limites de exposição/alavancagem; kill-switch.
- Entregáveis: módulo de risco integrado ao ambiente/executor.
- Gate: respeito a limites > 99% e redução de violações de risco.

## Fase 6 — UI/Dashboard (React)
- Componentes Recharts: `CandlestickChart`, `LineChart`, `AreaChart`; layout shadcn/ui + Tailwind; temas claro/escuro.
- Hooks: `useRealtimePrices`, `useCandles` integrando com backend/DB (ex.: Postgres/Supabase), estados `loading/error/success`.
- Métricas: painel interno com `PnL`, Sharpe, Drawdown, latência de atualização.
- Entregáveis: `frontend/` com componentes, hooks e testes.
- Gate: responsividade, acessibilidade (AA), performance em ≥ 50k pontos.

## Fase 7 — DevOps e Implantação
- Poetry: dependências fixadas (`pyproject.toml`), scripts de execução, `poetry run` universal.
- CI Windows: lint (`ruff`), tipos (`mypy`), testes (`pytest`), segurança (`safety`).
- Paper trading: serviço de inferência com monitoramento; go-live piloto com capital limitado e kill-switch.
- Entregáveis: `.github/workflows/ci.yml`, `.env.example`, scripts.
- Gate: 4–8 semanas de paper com métricas dentro dos limites e zero incidentes críticos.

## Fase 8 — QA/Testes
- Python (pytest): unitários e integração para `backtester`, `env`, `trainer`; `hypothesis` em métricas.
- Frontend (Vitest/RTL): charts, hooks real-time, acessibilidade e E2E (Playwright/Cypress).
- Coberturas mínimas: `>= 80%` módulos críticos; `>= 75%` hooks.
- Gate: suites verdes em CI com thresholds atendidos e ausência de flaky.

## Marcos e Critérios de Aceite
- Semana 1–2: Dados/Setup → `pytest` verde e pipelines prontos.
- Semana 3–4: Ambiente v1 + custos → PnL/custos consistente.
- Semana 5–6: Agentes baselines → superar baselines.
- Semana 7–8: HPO/aprimoramentos → metas de Sharpe/Sortino/MaxDD.
- Semana 9–10: Walk-forward/Stress → robustez confirmada.
- Semana 11–12: Paper trading → operação estável.
- Semana 13–16: Go-live piloto → decisão de escala.

## Ferramentas e Comandos
- Dependências (exemplos): `numpy`, `pandas`, `torch`, `gymnasium`, `optuna`, `pytest`, `python-dotenv`.
- Instalar: `poetry install`
- Testar: `poetry run pytest -q`
- Treinar: `poetry run python -m src.training.train --algo ppo --asset btc,xau --config configs/ppo.yaml`
- Backtest: `poetry run python -m src.evaluation.backtester --walkforward`
- Paper: `poetry run python -m src.execution.paper_trade`

## Riscos e Mitigações
- Custos subestimados → stress sistemático de `fees/slippage` e auditoria de execução.
- Vazamento de dados → revisões de splits e validações automáticas de índices temporais.
- Sobreajuste a regime específico → currículo e walk-forward multi‑regime; ablations.
- Falhas operacionais → kill-switch, circuit-breakers e observabilidade.

## KPIs de Sucesso
- `Sharpe ≥ 1.2`, `Sortino ≥ 1.5`, `PF ≥ 1.5`, `MDD ≤ 15%` (out‑of‑sample).
- Estabilidade em ≥ 3 regimes; sensibilidade baixa a custos.
- Uptime ≥ 99% em paper/live; respeito a limites > 99%.

---

# Roadmap Geral do Projeto — Versão Estendida

> Objetivo: consolidar um plano detalhado, incrementando funcionalidades e especificações para operação do bot em BTC e XAUUSD. Referências diretas ao código existente entre parênteses.

## Arquitetura
- Núcleo Python (Poetry) para ambiente, agentes e backtest (`pyproject.toml`).
- UI React/Vite para monitoramento e métricas (`src/`), com integração Supabase (`src/integrations/supabase`).
- Funções Supabase para persistência de episódios/trades (`supabase/functions/rl-training/index.ts`).
- Dados e artefatos: `data/`, `checkpoints/`, `results/`, `public/` (feeds e históricos de treino).

## Funcionalidades Prioritárias
- Ambiente OHLCV e Orderbook:
  - BTC OHLCV com custos/latência e risco (`python/environment/bitcoin_env.py:19`).
  - Orderbook sintético com spread e midprice (`python/environment/orderbook_env.py:14`).
  - Custos/Slippage consistentes (`python/utils/market_costs.py`).
- Backtesting e Métricas:
  - Motor de backtest com métricas financeiras e risco (`python/evaluation/backtester.py:16`).
  - Walk-forward e gráficos em `results/`.
- Agentes e Treinador:
  - Fábrica de agentes incluindo DQN, QRDQN, PPO, SAC, TD3 (`python/models/agent_factory.py:11`).
  - Treinador com checkpoints (`python/training/trainer.py`).
- Integração Supabase:
  - Função `rl-training` para episódios/trades e métricas (`supabase/functions/rl-training/index.ts:76`).
- UI e Observabilidade:
  - Componentes de métricas, trades e progresso (`src/components/*`).

## Expansões de Funcionalidade
- Dados
  - Feed em tempo real (WebSocket/REST) com resampling e reconciliação de gaps.
  - Versão de datasets e validação temporal (sem vazamento) com checks automáticos.
  - Engenharia de microestrutura: OFI, desequilíbrio, profundidade, filas.
  - Múltiplos ativos: BTC, XAUUSD; sincronização de fusos e sessões (23×5 para XAUUSD).

- Ambiente de Mercado
  - Tipos de ordem: market, limit, stop e cancelamentos; preenchimento parcial e prioridade preço/tempo.
  - Latência e slippage estocásticos com jitter de fee; escalonamento por volatilidade.
  - Restrições: alavancagem, limites de posição, kill-switch por max drawdown intradiário.
  - Espaço contínuo para alvo de posição e ordens limite (preço/quantidade).

- Recompensa e Risco
  - `risk-aware reward`: penalização de drawdown, turnover e inventário com `volatility targeting`.
  - `lambda_turn` para suavizar mudanças bruscas de posição.
  - `Kelly fraction` e `risk-parity` em portfólio BTC+XAUUSD (`python/utils/risk_manager.py`).

- Agentes
  - Distribucional (QR‑DQN) já disponível (`python/models/qrdqn_agent.py`).
  - Contínuos (SAC/TD3) via SB3, exigindo `spaces.Box` no ambiente (`python/models/agent_factory.py:91`).
  - PPO com extratores temporais (CNN/Transformer) (`python/models/feature_extractors.py`).
  - Ensemble e votação de políticas; calibração de confiança.

- Treinamento
  - `curriculum` por regimes de volatilidade; randomização de custos/latência.
  - HPO com Optuna; early stopping por métricas fora da amostra.
  - Checkpoints e resiliência a falhas; retomada segura.

- Avaliação
  - Métricas financeiras (Sharpe, Sortino, Calmar, PF), risco (VaR/CVaR), beta/alpha vs. benchmark.
  - Estresse de custos/latência; bootstrap de blocos e walk‑forward.
  - Logs de ordens para reconciliação e auditoria.

- Execução e Serviços
  - Serviço de `paper trading` com estados `IDLE/LEARNING/TRADING/ERROR`.
  - Endpoints internos para status, métricas e controle; autenticação.
  - Agendamento (Windows Task Scheduler) para rotinas de coleta e treino.
  - Circuit breakers, limites diários e reconciliação de posições.

- UI/Dashboard
  - Painéis de PnL, Sharpe, Drawdown, latência; charts de candles e retornos.
  - Hooks para dados em tempo real (Supabase/Postgres) com estados claros.
  - Responsividade, acessibilidade e temas claro/escuro.

- Observabilidade
  - Logs estruturados (JSON), rastreabilidade por `trade_id`, alertas (email/Discord).
  - Telemetria: latência de decisão, rejeições, fill ratio, divergência de slippage.

- Segurança
  - Segredos via `.env` e variáveis de ambiente; nunca em código.
  - Separação de roles no Supabase; chaves publishable no frontend (prefixo `VITE_`).

## Comandos Operacionais (Windows)
- Instalar: `poetry install`
- Testes: `poetry run pytest -q`
- Treino: `poetry run python python/main.py --mode train --data python/data/bitcoin_historical.csv`
- Backtest: `poetry run python python/main.py --mode backtest --data python/data/bitcoin_historical.csv --model checkpoints/best_model.pth`
- UI dev: `npm run dev`

## Roadmap por Fases
- Fase 0 — Setup e Qualidade
  - Seeds, padrões (PEP8, type hints), CI local, `.env`.
  - Gate: suíte de testes verde e qualidade estática OK.
- Fase 1 — Dados
  - Coleta, limpeza, microestrutura e partição temporal.
  - Gate: cobertura > 95% e validação sem vazamento.
- Fase 2 — Ambiente
  - Execução realista com ordens e custos; espaço contínuo opcional.
  - Gate: testes de sanidade de PnL/custos e reprodutibilidade.
- Fase 3 — Backtest
  - Métricas e walk‑forward; gráficos e relatórios.
  - Gate: estabilidade em múltiplos regimes.
- Fase 4 — Agentes
  - Baselines e RL (DQN/QRDQN/PPO/SAC/TD3); ensemble.
  - Gate: superar baselines fora da amostra.
- Fase 5 — Risco e Portfólio
  - Vol targeting, penalizações e limites; BTC+XAUUSD.
  - Gate: respeito a limites > 99%.
- Fase 6 — UI/Observabilidade
  - Painéis e alertas; telemetria.
  - Gate: performance e acessibilidade.
- Fase 7 — Operação
  - Paper trading 4–8 semanas; monitoramento e playbooks.
  - Gate: uptime e métricas dentro dos limites.

## Critérios de Aceite
- Métricas financeiras mínimas (Sharpe, Sortino, PF, MDD) atingidas fora da amostra.
- Robustez a custos/latência; ausência de `flaky tests`.
- Segurança de chaves e segregação de ambientes (`.env.*`).

## Referências ao Código
- Ambiente BTC: `python/environment/bitcoin_env.py:19`
- Ambiente Orderbook: `python/environment/orderbook_env.py:14`
- Custos/Slippage: `python/utils/market_costs.py:12`
- Risco: `python/utils/risk_manager.py:12`
- Backtester: `python/evaluation/backtester.py:16`
- Fábrica de agentes: `python/models/agent_factory.py:11`
- Função Supabase: `supabase/functions/rl-training/index.ts:76`


---

# Especificações Detalhadas e Funcionalidades Adicionais

## Arquitetura do Ambiente (Especificação)
- Observação (`Dict`):
  - `market_window: Box(T, F)` com `log_return`, `close`, `volume`, `spread`, `ATR`, `RSI`, `MACD`, `imbalance`, `time_features`.
  - `position: Box(1) ∈ [-1, 1]`, `cash: Box(1)`, `unrealized_pnl: Box(1)`.
  - Validações: shapes, NaN/Inf, monotonicidade temporal.
- Ações:
  - Discreto: `0=hold`, `1=buy`, `2=sell` (alvo ±1 com limite de mudança por passo).
  - Contínuo: `Box(1) ∈ [-1,1]` alvo de posição; executor converte em ordens respeitando `max_position_change`.
- Executor de Ordens:
  - `fees_bps`: maker/taker, impostos; `funding_rate` (perp) e `borrow_bps` (shorts).
  - `slippage_bps`: proporcional a `|Δpos|`, `spread`, `volatilidade`; modelos abaixo.
  - `latency_steps`: atrasos de execução; preenchimento parcial; cancelamentos.
  - Restrições: `max_leverage`, `position_limit`, `daily_loss_limit`, margens e stop-out.
- Recompensa (`risk-aware`):
  - Base: `dlogNAV_t - costs_t`.
  - Penalizações: `λ_vol*vol_t + λ_turn*|Δpos_t| + λ_dd*drawdown_t + λ_inv*|pos_t|`.
  - Bônus episódico: `Sharpe(window)`, `PF`, redução de `MDD`.
- API (`Gymnasium`): `reset(seed)`, `step(action)`, `render(mode)`, `close()`.

Exemplo de assinatura (simplificada):
```python
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym

class CryptoTradingEnv(gym.Env):
    def __init__(self, data: Dict[str, np.ndarray], fees_bps: float, slippage_bps: float, latency_steps: int = 0, action_space: str = "discrete"):
        ...
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        ...
    def step(self, action: int | float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        ...
```

## Modelos RL (Configs e Hiperparâmetros)
- DQN (discreto): `Double+Dueling`, `Huber loss`, `PER (α=0.6, β→1.0)`, `n-step=3`, `target soft τ=0.005`, `epsilon` decaindo 1.0→0.05.
- PPO (discreto/contínuo): `clip=0.2`, `GAE(λ=0.95)`, `γ=0.99`, `entropy_bonus=0.01`, `value_coef=0.5`, `lr=3e-4`, `mini_batches=4`, `epochs=10`.
- SAC (contínuo): `α` automático, `γ=0.99`, `τ=0.005`, `batch=256`, `replay=1e6`, `lr=3e-4`, `target entropy` por dimensão.
- TD3 (contínuo): `policy_delay=2`, `target_noise=0.2`, `noise_clip=0.5`, `γ=0.99`, `batch=256`, `lr=3e-4`.
- Arquiteturas: `CNN1D` para séries, `LSTM/GRU` para dependência temporal, `TransformerEncoder` leve (`n_layers=2`, `n_heads=4`, `d_model=256`).

## Recompensas (Formulações Práticas)
- `reward_t = log(NAV_t) - log(NAV_{t-1}) - costs_t - λ_vol*vol_t - λ_turn*|Δpos_t| - λ_dd*drawdown_t`.
- Proxy Sharpe de janela `W`: adicionar `+ k * mean(returns_{t-W:t}) / (std(returns_{t-W:t}) + ε)`.
- CVaR: penalizar perdas acima de `VaR_95` com peso `λ_cvar`.

## Modelos de Slippage
- Linear: `slippage_bps = a * |Δpos|`.
- Raiz quadrada (impacto de mercado): `slippage_bps = k * sqrt(|Δpos|)`.
- Volatilidade‑escala: `slippage_bps = k * |Δpos| * vol_t / vol_ref`.
- Spread‑aware: `slippage_bps = k1 * spread_t + k2 * |Δpos|`.

## Backtesting e Walk‑Forward
- Divisão temporal por blocos com re‑treino e validação; relatório por janela.
- Monte Carlo com bootstrap de blocos; perturbação de custos/latência.
- Stress de eventos: gaps, flash‑crash, baixa liquidez; validação de limites.
- Relatórios: equity curve, drawdown, distribuição de trades, custos acumulados, exposição, turnover.

## Métricas e Relatórios
- `Sharpe`, `Sortino`, `Calmar`, `MaxDD`, `PF`, `HitRate`, `AvgTrade`, `Turnover`, `Exposure`, `Capacity`.
- Significância: `deflated Sharpe`, `bootstrap p-value`, ajuste por múltiplas tentativas (HPO).
- Exportação: CSV/Parquet e figuras PNG/HTML interativas.

## DevOps e Segurança
- `pyproject.toml` com dependências bloqueadas; grupos dev para `pytest`, `mypy`, `ruff`, `safety`.
- `.env` não versionado: credenciais e URLs; mascarar segredos em logs.
- CI Windows: jobs de lint/tipos/testes/segurança; gates de cobertura `>= 80%`.

## Scripts CLI e Pastas
- `src/training/train.py`: `--algo`, `--asset`, `--config`, `--seed`.
- `src/evaluation/backtester.py`: `--agent`, `--walkforward`, `--stress-costs`.
- `src/execution/paper_trade.py`: `--risk-profile`, `--env`, `--kill-switch`.
- Estrutura sugerida: `src/`, `configs/`, `runs/`, `checkpoints/`, `tests/`, `frontend/`.

## Paper Trading e Live
- Serviço de inferência com estado (streaming), retries e reconciliação de ordens.
- Monitoramento: `PnL`, `risk`, `drift` de features, saúde do serviço; alertas.
- Retraining: gatilhos por degradação de métricas; versionamento de modelos.

## UI/Dashboard
- Recharts: `CandlestickChart`, `LineChart`, `AreaChart` com zoom/brush e tooltip avançado.
- shadcn/ui + Tailwind: `AppShell`, `Tabs`, `Cards`, `Skeleton`, temas claro/escuro.
- Hooks Supabase: `useRealtimePrices`, `useCandles` com paginação, caching e reconexão.
- Métricas: `MetricsCard` com latência, taxa de atualização, erros e reconexões.

## QA e Testes
- Python (pytest): unitários/integração para `env`, `backtester`, `trainer`; `hypothesis` para métricas.
- Frontend (Vitest/RTL): charts, hooks, acessibilidade; E2E com Playwright/Cypress.
- Coberturas: `>= 80%` módulos críticos; `>= 75%` hooks.

## Funcionalidades Futuras
- `Distributional RL (C51/QR-DQN)`, `Ensembles` e `Bootstrapped DQN` para incerteza.
- `Regime detection` e `policy switching` por contexto (volatilidade, tendência, liquidez).
- `Meta‑RL`/`offline RL` para aprendizado com dados históricos extensos.
- `Feature store` determinística e `drift detection` (KS/adfuller) integrada ao pipeline.

## Checklists de Aceite
- Dados: integridade, sem vazamento, cobertura > 95%.
- Ambiente: consistência contábil, custos e slippage realistas, seeds reprodutíveis.
- RL: superar baselines em validação; estabilidade de treino; checkpoints reproduzíveis.
- Backtest: robustez em regimes; stress tests; relatórios completos.
- UI: performance (≥ 50k pontos), acessibilidade AA, responsividade.
- DevOps: CI verde, segurança sem vulnerabilidades críticas.

---

# Templates de Configuração (YAML)

## Treino PPO (BTC e XAUUSD)
```yaml
env:
  name: "BTC_XAU_Env"
  assets: ["BTC", "XAUUSD"]
  timeframe: "5m"
  window_size: 128
  latency_steps: 1
  fees_bps: 5
  slippage_bps: 8
  max_leverage: 2.0
  position_limit: 1.0
  daily_loss_limit_bps: 200

reward:
  volatility_target: 0.10
  lambda_vol: 0.2
  lambda_turn: 0.05
  lambda_dd: 0.3
  lambda_inv: 0.02

algo:
  type: "ppo"
  clip: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coef: 0.01
  value_coef: 0.5
  lr: 0.0003
  num_epochs: 10
  mini_batches: 4
  cnn_channels: [32, 64]
  rnn_hidden: 256

training:
  total_steps: 1000000
  eval_every_episodes: 10
  eval_episodes: 10
  checkpoint_every_steps: 50000
  device: "cuda"
```

## Treino SAC (ação contínua)
```yaml
env:
  name: "BTC_XAU_Env"
  action_space: "continuous"
  timeframe: "1m"
  window_size: 64
  fees_bps: 7
  slippage_bps: 12

algo:
  type: "sac"
  gamma: 0.99
  tau: 0.005
  lr_actor: 0.0003
  lr_critic: 0.0003
  batch_size: 256
  replay_size: 1000000
  target_entropy: "auto"
  cnn_channels: [32, 64]
  transformer:
    layers: 2
    heads: 4
    d_model: 256
```

## Backtest Walk‑Forward
```yaml
backtest:
  windows:
    - { train: ["2018-01-01", "2020-12-31"], val: ["2021-01-01", "2021-06-30"], test: ["2021-07-01", "2021-12-31"] }
    - { train: ["2020-01-01", "2022-12-31"], val: ["2023-01-01", "2023-06-30"], test: ["2023-07-01", "2023-12-31"] }
  stress:
    costs_multiplier: [1.5, 2.0]
    latency_steps: [0, 1, 3]
    gap_events: true
metrics:
  sharpe_target: 1.2
  sortino_target: 1.5
  maxdd_limit: 0.15
```

---

# Esquemas de Dados (OHLCV e Livro de Ofertas)

## Candles (OHLCV)
- Colunas: `timestamp`, `symbol`, `timeframe`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `spread_bps`, `source`.
- Regras: timestamps UTC, sem duplicatas, `spread_bps` ≥ 0, `volume` ≥ 0.
- Derivados: `log_return`, `rolling_vol`, `ATR`, `EMA/SMA` multi‑janela.

## Livro de Ofertas (top‑N níveis)
- Colunas: `timestamp`, `symbol`, `bid_px_1..N`, `bid_qty_1..N`, `ask_px_1..N`, `ask_qty_1..N`.
- Sinais: `order_flow_imbalance`, `queue_imbalance`, `microprice`, `depth_ratio`.
- Qualidade: remoção de `bad ticks`, reconstrução de `midprice` e `spread`.

---

# Tipos de Ordem e Execução
- Market, Limit, IOC, FOK, Stop, Trailing Stop.
- Preenchimento parcial e cancelamentos com prioridade preço/tempo.
- Price improvement e proteção de preço em alta volatilidade.
- Reconciliação: ledger de ordens, execuções, custos e PnL realizado.

---

# Gestão de Risco Avançada
- Volatility Targeting: ajustar posição para atingir volatilidade alvo diária/semanal.
- VaR/CVaR contínuos: estimativa por janela e bloqueios quando excedidos.
- Limites: `gross/net exposure`, `leverage`, `position_limit`, `daily_loss_limit`, `kill_switch` por `MaxDD`.
- Paridade de risco BTC/XAU: alocação por contribuição de risco; rebalance periódico.

---

# MLOps e Observabilidade
- Versionamento: SemVer de modelos, fingerprint dos dados, registro de artefatos.
- Drift: testes KS/ADFuller e alarmes; re‑treino gatilhado por deterioração.
- Logs: estruturados, sem segredos; tracing básico de decisões e execuções.
- Dashboards: `PnL`, risco, uptime, taxa de erro, latência e slippage.

---

# Playbook de Paper e Live
- Paper (4–8 semanas): validar métricas, slippage e estabilidade; simular falhas de rede.
- Pré‑trade: checagem de limites, margem, posição, horário/sessão (XAUUSD 23×5).
- Pós‑trade: reconciliação, auditoria de custos/execuções, update de risco.
- Go‑live piloto: capital limitado, monitoramento e `kill_switch` ativo.

---

# UI/Dashboard — Funcionalidades Adicionais
- Multi‑símbolo e troca de `timeframe` com caching e downsampling.
- Modo offline com re‑play de dados históricos; exportação CSV/PNG.
- Painéis: `Equity`, `Drawdown`, `Trades`, `Custos`, `Exposição`; filtros e buscas.
- Acessibilidade: navegação por teclado, foco visível, contraste AA; temas claro/escuro.

---

# QA — Cenários e Critérios
- Edge cases: zero volatilidade, gaps, buracos de dados, spreads extremos, baixa liquidez.
- Property‑based: invariantes de métricas e contabilidade; reconciliação de PnL.
- Determinismo: seeds replicáveis; ausência de flaky; cobertura `>= 80%`.

---

# Orçamentos de Performance
- Backtest: tempo por janela e uso de memória estáveis; datasets ≥ 1e6 linhas.
- Treino: throughput com GPU; evitar thrash CPU↔GPU; batches otimizados.
- UI: render com ≥ 50k pontos e FPS ≥ 50; LCP ≤ 2.5s; CLS ≤ 0.1.

---

# Cronograma Detalhado
- Semana 1–2: Setup, dados e validações; CI e qualidade estática.
- Semana 3–4: Ambiente v1 com custos e latência; testes de sanidade.
- Semana 5–6: Baselines RL (DQN/PPO) e comparação com buy‑and‑hold/momentum.
- Semana 7–8: HPO (Optuna), LSTM/Transformer leve; robustez e ablations.
- Semana 9–10: Walk‑forward, Monte Carlo e stress; relatório completo.
- Semana 11–12: Paper trading; monitoramento e risco online.
- Semana 13–16: Go‑live piloto; retraining automático; auditoria e decisão de escala.

---

# Comandos Úteis (Poetry, Windows)
- Instalar: `poetry install`
- Testes: `poetry run pytest -q`
- Lint/tipos: `poetry run ruff check .` e `poetry run mypy .`
- Treino PPO: `poetry run python -m src.training.train --algo ppo --asset btc,xau --config configs/ppo.yaml`
- Treino SAC: `poetry run python -m src.training.train --algo sac --asset btc,xau --config configs/sac.yaml`
- Backtest: `poetry run python -m src.evaluation.backtester --walkforward --config configs/backtest.yaml`
- Paper: `poetry run python -m src.execution.paper_trade --env .env --risk-profile conservative`

