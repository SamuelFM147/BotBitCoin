# Histórico de Mudanças – Sistema de Recompensas (BTC/XAUUSD)

> Versão 1.0 – Data: 2025-11-11

## Diagnóstico Inicial
- Recompensa baseada em `value_change` normalizado por `initial_balance` em `python/environment/bitcoin_env.py:142–154`.
- Custos e slippage aplicados à balança/posição; penalidade de custo desativada por padrão.
- Sem `volatility scaling` (`σ_target`) e sem penalização incremental de drawdown na recompensa.
- Métricas robustas em `python/evaluation/backtester.py:105–178`, risco em `python/utils/risk_manager.py:179–219`.

## Mudanças Implementadas
- `Volatility scaling` com `σ_target` por janela móvel (`vol_window`) e piso `sigma_floor`.
- Penalização incremental de drawdown: `λ_dd * max(0, DD_t - DD_{t-1})`.
- `Shaping` por custos e turnover: custos por passo derivados de `fee` e `slippage_bps`; penalidade de `λ_turn * |Δpos_t|`.
- Penalidade de inventário opcional: `λ_inv * |pos_ratio_t|`.
- `reward_include_fee_penalty` ativado por padrão.

### Arquivos e Linhas
- Ambiente:
  - `python/environment/bitcoin_env.py:28–44` – novos parâmetros do construtor (`reward_include_fee_penalty`, `vol_window`, `sigma_floor`, `lambda_dd`, `lambda_inv`, `lambda_turn`).
  - `python/environment/bitcoin_env.py:53–60` – armazenamento de parâmetros.
  - `python/environment/bitcoin_env.py:84–87` – estado para drawdown e posição anterior.
  - `python/environment/bitcoin_env.py:99–103` – reset de estados auxiliares.
  - `python/environment/bitcoin_env.py:132–156` – refatorado para `volatility scaling`, penalização de drawdown, inventário e turnover; extração de custo por `fee` e `slippage_bps`.
  - `python/environment/bitcoin_env.py:158–171` – captura de `delta_position` e uso na recompensa.
  - `python/environment/bitcoin_env.py:289–300` – `info` expandido com `sigma_target`, `drawdown`, `dd_delta`, `turnover`, `cost_t`.
- Configuração:
  - `python/config/config.py:13–18` – novos campos em `EnvironmentConfig`: `reward_include_fee_penalty`, `vol_window`, `sigma_floor`, `lambda_dd`, `lambda_inv`, `lambda_turn`.

## Fórmula de Recompensa Atualizada
`r_t = reward_scaling * \frac{(\Delta \text{Valor}_t / \text{Inicial}) - cost_t - risk_t - λ_{dd}\cdot \Delta DD_t - λ_{turn}\cdot |\Delta pos_t|}{\sigma_{target}}`

Onde:
- `cost_t` usa `fee` + custo de `slippage_bps` do último trade quando há ação.
- `risk_t` inclui penalização por exceder `max_position_size` e `λ_inv * |pos_ratio_t|`.
- `\sigma_{target}` é o desvio-padrão dos retornos de preço na janela `vol_window` com piso `sigma_floor`.
- `\Delta DD_t` é o incremento de drawdown via `RiskManager.update_drawdown`.

## Justificativas
- Estabilidade: `volatility scaling` reduz variação de gradiente em regimes distintos.
- Robustez: penalização incremental de drawdown e custos/turnover alinham a política a risco e fricções reais.
- Rentabilidade líquida: internalização de `fees/slippage` e `turnover` reduz churn.

## Impacto Esperado em Métricas
- `Sharpe` e `Sortino` mais estáveis; redução de `Max Drawdown` e `Downside deviation`.
- `Win rate` potencialmente menor, mas `Profit Factor` maior devido a menor custo efetivo por trade.

## Requisitos/Parâmetros
- `vol_window` padrão: 50; `sigma_floor`: `1e-6`.
- `λ_dd` padrão: `0.1`; `λ_inv` e `λ_turn` padrão: `0.0`.
- `reward_include_fee_penalty=True` por padrão.

## Referências
- Oxford‑Man Institute, Journal of Financial Data Science – `volatility scaling` em multi-futuros.
- Denny Britz (2018) – POMDP e desenho de recompensas em trading.
- arXiv 2311.02088 – Order book + TD‑RL com ênfase em custos/slippage/spread (inclui XAUUSD).

## Próximos Passos
- Métricas expandidas no backtester: `Profit Factor`, `Turnover`, `Exposure`, `avg_fee_per_trade`, `avg_slippage_bps`, `alpha/beta vs benchmark`, `sharpe_stability`.
- Automação de `walk‑forward` adicionada em `Backtester.walk_forward` com avaliação por janelas rolantes.
 - Adição de QR‑DQN como alternativa de agente (`python/models/qrdqn_agent.py`) e seleção via `--algo qrdqn` no CLI.
 - Robustez extra: `reward_clip_abs` e `fee_jitter_pct` na configuração de ambiente para clipping de recompensa e aleatorização de custos.
