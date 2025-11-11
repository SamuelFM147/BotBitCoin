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

