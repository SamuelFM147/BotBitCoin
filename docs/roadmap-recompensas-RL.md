# Roadmap de Recompensas para Agentes de RL em Trading

## Visão Geral
Este documento define um roadmap prático, incremental e robusto para projetar, implementar, testar e evoluir funções de recompensa para agentes de aprendizado por reforço (RL) aplicados a trading (ex.: Bitcoin/XAUUSD). Foca em recompensa densa (passo a passo), componentes de risco, técnicas de reward shaping, avaliação realista e segurança, seguindo boas práticas de engenharia em Python (PEP 8, type hints, testes com `pytest`) e execução via `poetry run` em Windows.

## Objetivos
- Maximizar lucro sustentável ajustado ao risco em diferentes regimes de mercado.
- Controlar drawdown e exposição, reduzindo rotatividade e custos de transação.
- Garantir robustez (out-of-sample), evitar reward hacking e promover segurança.
- Facilitar observabilidade com breakdown por componente e métricas de sucesso claras.

## Pré-requisitos e Setup
- Gerenciador de pacotes: Poetry.
  - `poetry install`
  - `poetry run pytest`
  - `poetry run python scripts/ablation_rewards.py`
- Testes unitários com `pytest` em `tests/`.
- Variáveis sensíveis em `.env` (não versionado); nunca usar informações futuras.

## Glossário
- `PnL realizado`: Lucro/prejuízo efetivo no passo, após execução.
- `Custos`: Taxas, spread, slippage e impacto de mercado.
- `Exposição`: Valor absoluto da posição (normalizado por capital).
- `Rotatividade (turnover)`: Magnitude de mudança de posição.
- `Drawdown`: Queda do pico histórico de capital.
- `Regime`: Caracterização do ambiente (volatilidade, tendência, liquidez).

## Arquitetura da Recompensa
Propõe-se uma engine modular com componentes configuráveis e escaláveis:

### Componentes do Reward
- `pnl_component`: PnL realizado por passo.
- `transaction_costs`: custos e penalidade por cruzar spread.
- `position_change_penalty`: penaliza |Δposição| para reduzir overtrading.
- `exposure_penalty`: penaliza exposição excessiva (inventário, overnight se aplicável).
- `drawdown_penalty`: penaliza drawdown corrente ou incremental.
- `leverage_penalty`: penaliza alavancagem acima de limites.
- `volatility_scaling`: ajusta magnitude do reward por regime de volatilidade.
- `potential_shaping`: `γ·Φ(s') − Φ(s)` com potenciais seguros (balance, risk buffer).
- `episodic_bonus`: bônus por métrica episódica (Sharpe/Sortino) ao final.

### Especificação de API (proposta)
```python
from typing import Dict, Optional

class RewardEngine:
    def __init__(
        self,
        weights: Dict[str, float],
        clip_range: Optional[float] = 5.0,
        use_volatility_scaling: bool = True,
        potential_fn: Optional[callable] = None,
    ) -> None:
        """Engine de recompensas modular.
        Parâmetros:
        - weights: pesos para componentes (ex.: {"pnl":1.0, "costs":0.5, ...}).
        - clip_range: limite de clipping para estabilidade do treino.
        - use_volatility_scaling: ativa ajuste por regime.
        - potential_fn: Φ(s) para shaping potencial (opcional).
        """

    def step(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        exec_report: Dict,
    ) -> Dict[str, float]:
        """Computa reward por passo e retorna breakdown por componente."""

    def episode_finalize(self, episode_metrics: Dict[str, float]) -> float:
        """Aplica bônus episódico (ex.: Sharpe/Sortino) se configurado."""
```

## Fórmulas Base
### Reward Denso por Passo
```
r_t = w_pnl·pnl_realizado_t
      − w_costs·custos_t
      − w_poschg·|Δposição_t|
      − w_expo·|exposição_t|
      − w_dd·drawdown_t
      − w_lev·excesso_alavancagem_t
```
Com `volatility_scaling`: `r_t ← r_t / (ε + vol_regime_t)` e `clipping`: `r_t ← clip(r_t, −C, +C)`.

### Potential-Based Shaping (seguro)
```
r'_t = r_t + γ·Φ(s_{t+1}) − Φ(s_t)
```
Sugestões de `Φ(s)`: `balance_norm`, `risk_buffer`, `unrealized_pnl_norm` (sem lookahead).

### Bônus Episódico (ex.: Sharpe/Sortino)
```
R_ep = w_sharpe·Sharpe_ep + w_sortino·Sortino_ep
       − w_mdd·MaxDrawdown_ep − w_turn·Turnover_ep
```

## Pesos e Agendamento (Curriculum)
- Fase 1 (aquecimento): `w_pnl=1.0`, `w_costs=0.2`, `w_poschg=0.1`, demais ~0.
- Fase 2 (controle de risco): aumentar `w_expo` e `w_dd` gradualmente.
- Fase 3 (robustez): ativar `volatility_scaling` e `potential_shaping`.
- Fase 4 (refino): introduzir `episodic_bonus` e ajustar `w_mdd`, `w_turn`.
Agendar pesos por passos/episódios ou por detecção de regime.

## Normalização e Estabilidade
- Normalizar entradas (pnl, custos, exposição) por capital, ATR ou volatilidade.
- Aplicar `tanh`/`z-score` por regime quando necessário.
- Clipping conservador para evitar gradientes explosivos.

## Multiobjetivo e Scalarização
- Tratar lucro, custo, risco e rotatividade como objetivos simultâneos.
- Usar combinação linear com pesos e, opcionalmente, `Pareto front` para análise offline.
- Agendar pesos (curriculum) e testar sensibilidade por ablação.

## Segurança e Conformidade
- Penalidades fortes para violações de limites (exposição, alavancagem, risco por instrumento).
- Sem informações futuras; recompensa somente após execução confirmada (`exec_report`).
- Hard caps de risco na própria environment (constrained RL) além das penalidades.

## Anti-Reward Hacking
- Monitorar discrepâncias entre `pnl_realizado` e métricas de risco.
- Alertar quando reward é alto com `turnover` extremo ou slippage anormal.
- Rodar testes de regime e adversariais simples (stress de volatilidade).

## Logging e Telemetria
- Registrar breakdown por componente: `{"pnl":..., "costs":..., "expo":..., ...}`.
- Log de `episodic_bonus` e métricas: Sharpe, Sortino, MaxDD, Profit Factor.
- Traçar séries (Recharts/Matplotlib) e armazenar em Supabase/CSV conforme o projeto.

## Avaliação e Backtesting
- Backtests com custos e slippage realistas; simular fila/impacto.
- Validação `out-of-sample`, `regime randomization` e `walk-forward`.
- Ablação: remover um componente por vez e medir efeito em métricas.

## Testes (pytest)
- Unitários: cada componente (custos, drawdown, exposição) e clipping.
- Integração: episódio completo com `episodic_bonus` e verificação de logs.
- Robustez: cenários de alta/baixa volatilidade com resultados esperados.

## Fluxo de Iteração
1) Implementar RewardEngine com componentes essenciais.
2) Habilitar `volatility_scaling` e `potential_shaping` com Φ(s) seguro.
3) Adicionar `episodic_bonus` (Sharpe/Sortino) e testes correspondentes.
4) Executar ablação e ajustar pesos; monitorar estabilidade de treinamento.
5) Validar em backtests e regimes distintos; revisar pesos e limites.

## Funcionalidades Avançadas
- Detecção de regime online (volatilidade, tendência) com ajuste automático de pesos.
- Ajuste por `CVaR/Expected Shortfall` no reward (episódico ou janela móvel).
- Penalidade de `Ulcer Index` para suavidade de curva de capital.
- `Risk of ruin` como penalidade crescente ao se aproximar de limites.
- `Adaptive turnover` (penalidade maior em spreads largos/baixa liquidez).

## Planos de Experimentação
- Random/Grid search de pesos `{w_pnl, w_costs, w_expo, w_dd, w_poschg}`.
- Comparação com e sem `volatility_scaling` e `potential_shaping`.
- Avaliar impacto de `episodic_bonus` nas métricas out-of-sample.

## Métricas de Sucesso
- Desempenho: PnL, Sharpe, Sortino, Profit Factor.
- Risco: Max Drawdown, CVaR, Ulcer Index, exposição média.
- Comportamento: turnover, holding time, slippage médio, contribuição por componente.
- Robustez: estabilidade out-of-sample, resiliência a regimes, sensibilidade a custos.

## Exemplo de Configuração (YAML/JSON)
```yaml
weights:
  pnl: 1.0
  costs: 0.3
  poschg: 0.15
  expo: 0.25
  dd: 0.2
  lev: 0.1
episodic_bonus:
  sharpe: 0.5
  sortino: 0.3
  mdd: 0.4
  turn: 0.2
clip_range: 5.0
use_volatility_scaling: true
potential_shaping: true
```

## Scripts Sugeridos
- `scripts/ablation_rewards.py`: roda ablação variando pesos e grava métricas.
- `evaluation/backtester.py`: backtest com custos/slippage, walk-forward e relatórios.

## Checklist de Produção
- Limites de exposição/alavancagem aplicados e testados.
- Logs de breakdown e métricas persistidos.
- Validação out-of-sample e stress tests concluídos.
- Monitoramento para reward hacking e alertas configurados.

## Roadmap de Evolução (Resumo)
- Curto prazo: Reward denso com custos, exposição e drawdown + logging.
- Médio prazo: Volatility scaling, potential-based shaping e bônus episódico.
- Longo prazo: Regime-aware adaptativo, CVaR e constrained RL com monitoramento avançado.

## Modelos de Custos e Slippage
- Custos fixos e proporcionais: `custos_t = fee_fixed + fee_rate·|notional_t|`.
- Spread e crossing: penalizar ordens que cruzam o spread; `spread_cost_t = 0.5·spread_t·|Δposição_t|`.
- Slippage por impacto: modelo raiz-quadrada `slip_t = k·σ·√(|volume_t|)` ou impacto linear com coeficiente calibrado.
- Execução: separar `ordens limite` e `ordens a mercado` com custos distintos; incorporar rejeições e filas.

## Detecção de Regimes (Adaptativa)
- Volatilidade realizada: `σ_t` por janela móvel (ex.: ATR, desvio padrão log-retornos).
- Tendência: alvos de `EMA/SMMA` cruzados, slope de regressão linear de preço.
- Liquidez: profundidade de livro e volume; penalidades maiores em baixa liquidez.
- Classificador simples: regras por limiares (`vol alta/baixa`, `trend up/down/flat`) para ajustar pesos.

## Definições Matemáticas (Métricas)
- Sharpe: `Sharpe = (E[R] - R_f) / σ(R)`; usar `R_f≈0` em cripto.
- Sortino: `Sortino = (E[R] - R_f) / σ_-(R)` com desvio apenas de perdas.
- Max Drawdown: `MDD = max_{t}(Peak_t - Equity_t) / Peak_t`.
- CVaR (nível α): média das perdas além do VaR; `CVaR_α = E[Loss | Loss ≥ VaR_α]`.
- Ulcer Index: raiz da média dos quadrados de drawdowns.

## Portfolio e Multi-Agente
- Portfolio-level reward: somar PnL por ativo com penalidades de correlação e concentração.
- Penalizar concentração: `concentration_penalty = λ·∑ w_i^2`.
- Multi-agente: agentes por ativo com `coordinator` que distribui `episodic_bonus` pelo conjunto; evitar competição destrutiva via limites globais de risco.

## Safe RL e Constrained RL
- Lagrangiano: otimizar `J(π) − λ·E[g(x)]` com atualização de `λ` para satisfazer `E[g(x)]≤c` (ex.: exposição média).
- CPO (Constrained Policy Optimization) e Lyapunov-based: garantir limites de segurança por passo/episódio.
- Integrar hard caps na environment e penalidades no reward.

## Pesos Adaptativos (Controladores)
- PID sobre métricas de risco: ajustar `w_dd`/`w_expo` dinamicamente para manter `MDD` e exposição dentro de metas.
- Scheduler por regime: pesos diferentes quando `vol alta` vs `vol baixa`.
- `Adaptive turnover`: aumentar `w_poschg` quando spread e slippage se elevam.

## Observabilidade: Esquema de Logs
```json
{
  "step": 1245,
  "reward": {
    "total": 0.38,
    "pnl": 0.72,
    "costs": -0.21,
    "poschg": -0.05,
    "expo": -0.04,
    "dd": -0.04,
    "lev": 0.00
  },
  "metrics": {
    "vol_regime": 0.012,
    "spread": 0.50,
    "turnover": 0.18
  }
}
```
- Persistir logs em JSON/CSV, com índices de episódio e timestamps.
- Traçar contribuição de cada componente ao longo do treino.

## Pipeline de Dados e Execução
- Dados OHLCV/ticks com sincronização e limpeza (sem buracos/duplicatas).
- Simulador de execução com latência e fila; sem uso de informações futuras.
- Normalização por capital e volatilidade; reset consistente entre episódios.

## CLI e Uso (Poetry)
- Instalação: `poetry install`.
- Testes: `poetry run pytest -q`.
- Execução de experimentos: `poetry run python scripts/ablation_rewards.py --config configs/reward.yaml`.
- Backtest: `poetry run python evaluation/backtester.py --config configs/backtest.yaml`.

## Troubleshooting e Anti-Hacking
- Reward explode/clipe constante: revisar normalização e `clip_range`.
- Alto Sharpe com turnover extremo: aumentar `w_poschg`, revisar custos/slippage.
- PnL bom mas MDD alto: elevar `w_dd`, usar `episodic_bonus` com penalidade de MDD.
- Drift entre treino e teste: reforçar `regime randomization` e walk-forward.

## Microestrutura (Extensões)
- Modelo de fila: estimar probabilidade de execução de ordens limite.
- Impacto temporário vs permanente: penalidades diferenciadas por tipo de impacto.
- `Queue position` e cancelamentos: afetar custos e `poschg` conforme rejeições.

## Pseudocódigo da RewardEngine (Exemplo)
```python
def compute_reward(state, action, next_state, exec):
    pnl = exec.get("realized_pnl", 0.0)
    costs = exec.get("fees", 0.0) + 0.5 * state["spread"] * abs(action["delta_pos"]) + exec.get("slippage", 0.0)
    poschg = abs(action["delta_pos"]) 
    expo = abs(next_state["position"]) / max(next_state.get("capital", 1.0), 1e-9)
    dd = next_state.get("drawdown", 0.0)
    lev_excess = max(0.0, next_state.get("leverage", 0.0) - next_state.get("leverage_limit", 1.0))

    r = (w_pnl * pnl - w_costs * costs - w_poschg * poschg - w_expo * expo - w_dd * dd - w_lev * lev_excess)
    if use_vol_scaling:
        r = r / (1e-6 + next_state.get("vol_regime", 1.0))
    if clip_range:
        r = max(-clip_range, min(clip_range, r))

    if potential_fn is not None:
        r += gamma * potential_fn(next_state) - potential_fn(state)

    return {
        "total": r,
        "pnl": w_pnl * pnl,
        "costs": - w_costs * costs,
        "poschg": - w_poschg * poschg,
        "expo": - w_expo * expo,
        "dd": - w_dd * dd,
        "lev": - w_lev * lev_excess,
    }
```

## Cronograma de Experimentos
- Semana 1: Implementar componentes base, testes unitários e logs.
- Semana 2: Volatility scaling, potential shaping e ablação de pesos.
- Semana 3: Bônus episódico, walk-forward e stress de regimes.
- Semana 4: Safe RL (Lagrangiano), PID de pesos e relatório final.