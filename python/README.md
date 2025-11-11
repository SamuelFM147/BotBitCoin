# Bitcoin RL Trading System

Sistema completo de trading automatizado para Bitcoin utilizando Reinforcement Learning (Deep Q-Network).

## 🎯 Características

- **Ambiente de Trading Customizado**: Simulador realista de mercado Bitcoin
- **Agente DQN**: Deep Q-Network com replay buffer e target network
- **Pipeline de ML Completo**: Pré-processamento, feature engineering e normalização
- **Sistema de Backtesting**: Avaliação completa com métricas financeiras
- **Gerenciamento de Risco**: Stop-loss, position sizing e drawdown control
- **Monitoramento**: Logs detalhados e visualizações

## 📋 Requisitos

### Hardware Recomendado
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA com CUDA (opcional mas recomendado)
- **Storage**: 10GB+ livre

### Software
- Python 3.8+
- CUDA 11.x (opcional, para GPU)

## 🚀 Instalação

1. **Clone o repositório**
```bash
cd python
```

2. **Crie ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale dependências**
```bash
pip install -r requirements.txt
```

### Alternativa Recomendada: Poetry

Você pode gerenciar dependências com Poetry para maior reprodutibilidade e isolamento de ambiente.

1. Instale o Poetry (Windows/Linux/Mac):
```bash
# Recomendado: via pipx
python -m pip install --user pipx
pipx install poetry

# Alternativa: instalador oficial
curl -sSL https://install.python-poetry.org | python -
```

2. Instale o ambiente e dependências:
```bash
poetry install
```

3. Execute comandos no ambiente Poetry:
```bash
# Treinar
poetry run python python/main.py --mode train --data data/bitcoin_historical.csv

# Backtest
poetry run python python/main.py --mode backtest --data data/bitcoin_test.csv --model checkpoints/best_model.pth

# Testes
poetry run pytest -q
```

4. Exportar `requirements.txt` para compatibilidade (opcional):
```bash
poetry export -f requirements.txt --output python/requirements.txt --without-hashes
```

Notas:
- Por padrão, o Poetry usa um venv fora do projeto. Se preferir `.venv` dentro do repo, habilite:
```bash
poetry config virtualenvs.in-project true
```
- Para usar PyTorch com GPU, instale a variante CUDA conforme sua versão:
```bash
# Exemplo CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Preparação de Dados

### Formato dos Dados

Os dados devem estar em CSV com as seguintes colunas:
- `timestamp`: Data/hora (formato ISO)
- `open`: Preço de abertura
- `high`: Preço máximo
- `low`: Preço mínimo
- `close`: Preço de fechamento
- `volume`: Volume negociado

### Exemplo de Download de Dados

```python
import ccxt
import pandas as pd

# Conectar à exchange
exchange = ccxt.binance()

# Baixar dados históricos
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=10000)

# Converter para DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Salvar
df.to_csv('data/bitcoin_historical.csv', index=False)
```

## 🎓 Treinamento

### Treinamento Básico

```bash
python main.py --mode train --data data/bitcoin_historical.csv
```

### Treinamento com Configuração Customizada

1. Crie arquivo `config.yaml`:
```yaml
environment:
  initial_balance: 10000.0
  max_position_size: 0.3
  transaction_cost: 0.001
  lookback_window: 50

dqn:
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  batch_size: 64
  hidden_layers: [256, 256, 128]

training:
  total_episodes: 5000
  eval_frequency: 100
  checkpoint_frequency: 500
```

2. Execute:
```bash
python main.py --mode train --data data/bitcoin_historical.csv --config config.yaml
```

### Monitoramento do Treinamento

Os logs são salvos em:
- `logs/training_history.json`: Histórico completo
- `checkpoints/`: Checkpoints do modelo
- TensorBoard (opcional):
```bash
tensorboard --logdir=logs/
```

## 📈 Backtesting

### Executar Backtest

```bash
python main.py --mode backtest \
  --data data/bitcoin_test.csv \
  --model checkpoints/best_model.pth
```

### Métricas Disponíveis

O backtest calcula:
- **Total Return**: Retorno total em %
- **Sharpe Ratio**: Retorno ajustado ao risco
- **Maximum Drawdown**: Maior perda acumulada
- **Win Rate**: Porcentagem de trades lucrativos
- **Sortino Ratio**: Retorno/volatilidade negativa
- **Calmar Ratio**: Retorno/máximo drawdown

## 📁 Estrutura do Projeto

```
python/
├── config/
│   └── config.py              # Gerenciamento de configurações
├── data/
│   ├── preprocessor.py        # Pipeline de pré-processamento
│   └── feature_engineer.py    # Engenharia de features
├── environment/
│   └── bitcoin_env.py         # Ambiente de trading
├── models/
│   ├── dqn_agent.py          # Agente DQN
│   └── ppo_agent.py          # Agente PPO (futuro)
├── training/
│   └── trainer.py            # Loop de treinamento
├── evaluation/
│   ├── backtester.py         # Sistema de backtesting
│   └── metrics.py            # Métricas de avaliação
├── utils/
│   └── risk_manager.py       # Gerenciamento de risco
├── main.py                   # Script principal
├── requirements.txt          # Dependências
└── README.md                 # Esta documentação
```

## 🔧 Módulos Principais

### 1. Environment (bitcoin_env.py)

Ambiente de trading compatível com OpenAI Gym:

```python
from environment.bitcoin_env import BitcoinTradingEnv

env = BitcoinTradingEnv(
    df=data,
    initial_balance=10000,
    lookback_window=50
)

obs, info = env.reset()
action = agent.select_action(obs)
next_obs, reward, done, truncated, info = env.step(action)
```

### 2. DQN Agent (dqn_agent.py)

Agente com Experience Replay e Target Network:

```python
from models.dqn_agent import DQNAgent

agent = DQNAgent(
    state_dim=obs_space.shape[0],
    action_dim=action_space.n,
    learning_rate=0.0001
)

# Selecionar ação
action = agent.select_action(state)

# Treinar
agent.store_transition(state, action, reward, next_state, done)
loss = agent.train()
```

### 3. Feature Engineering (feature_engineer.py)

Mais de 50 indicadores técnicos:

```python
from data.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.engineer_features(df)

# Features incluem:
# - Momentum: RSI, MACD, Stochastic
# - Trend: SMAs, EMAs, ADX
# - Volatility: Bollinger Bands, ATR
# - Volume: Volume ratios e médias
```

### 4. Risk Management (risk_manager.py)

Controle de risco integrado:

```python
from utils.risk_manager import RiskManager

risk_mgr = RiskManager(
    max_position_size=0.3,
    max_drawdown_limit=0.20,
    stop_loss_pct=0.05
)

# Verificar tamanho de posição
is_valid = risk_mgr.check_position_size(position_value, total_value)

# Calcular Kelly Criterion
kelly_size = risk_mgr.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
```

## 🎯 Uso Avançado

### Treinamento com GPU

```python
# Certifique-se de ter PyTorch com CUDA instalado
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O agente detecta automaticamente GPU disponível
agent = DQNAgent(..., device='cuda')
```

### Hyperparameter Tuning

Use Optuna para otimização automática:

```python
import optuna

def objective(trial):
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.99)
    
    agent = DQNAgent(state_dim, action_dim, 
                    learning_rate=learning_rate,
                    gamma=gamma)
    
    # Treinar e retornar métrica
    trainer = Trainer(agent, env)
    history = trainer.train()
    return max(history['eval_rewards'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Ensemble de Modelos

```python
# Treinar múltiplos agentes
agents = []
for i in range(5):
    agent = DQNAgent(state_dim, action_dim)
    trainer = Trainer(agent, env)
    trainer.train()
    agents.append(agent)

# Votar nas ações
def ensemble_action(state, agents):
    votes = [agent.select_action(state, training=False) for agent in agents]
    return max(set(votes), key=votes.count)
```

## 📊 Resultados Esperados

Com configuração padrão e ~2 anos de dados horários:

- **Training Time**: 2-6 horas (GPU) / 8-24 horas (CPU)
- **Win Rate**: 55-65%
- **Sharpe Ratio**: 1.5-2.5
- **Max Drawdown**: 10-20%
- **Annual Return**: 30-80% (backtest)

⚠️ **IMPORTANTE**: Resultados passados não garantem resultados futuros. Trading de criptomoedas envolve riscos significativos.

## 🐛 Troubleshooting

### Erro: CUDA out of memory
```bash
# Reduza batch_size e hidden layers
batch_size: 32
hidden_layers: [128, 128]
```

### Erro: No module named 'ta'
```bash
pip install ta
```

### Agent não aprende
- Verifique se os dados estão normalizados
- Aumente epsilon_decay para mais exploração
- Reduza learning_rate
- Aumente buffer_size

## 📚 Referências

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

## 📝 Licença

MIT License - veja LICENSE para detalhes

## 👥 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## ⚠️ Disclaimer

Este sistema é para fins educacionais e de pesquisa. Não constitui aconselhamento financeiro. Trading de criptomoedas envolve riscos significativos de perda. Use por sua conta e risco.

---

## 🔌 Integração com Supabase (episódios e trades)

Para persistir dados de treinamento em tempo real no Supabase usando a Edge Function `rl-training`:

- Configure variáveis de ambiente no arquivo `.env` na raiz (compartilhadas com o frontend):

```
VITE_SUPABASE_URL="https://<project-ref>.supabase.co"
VITE_SUPABASE_PUBLISHABLE_KEY="<anon-key>"
```

- O cliente Python deriva automaticamente a URL da função: `https://<project-ref>.functions.supabase.co/rl-training`.

- Uso no `Trainer`:

```python
from integrations.supabase_client import SupabaseEdgeClient
from training.trainer import Trainer

supabase = SupabaseEdgeClient()  # lê .env
trainer = Trainer(agent, env, eval_env, supabase_client=supabase, agent_id="DQN-v2.1")
history = trainer.train()
```

- Episódios persistidos: `agent_id`, `episode_number`, `total_reward`, `avg_loss`, `epsilon`, `actions_taken`, `duration_seconds`
- Trades persistidos: `agent_id`, `episode_id`, `trade_type`, `price`, `amount`, `profit_loss`, `confidence (opcional)`

Resiliência:
- Chamadas usam retries e logs em falhas; o loop de treinamento continua mesmo sem conexão.
- Para desativar temporariamente os retries, defina `SUPABASE_DISABLE_RETRIES="true"` no `.env`.
- Para ajustar explicitamente o número de tentativas, use `SUPABASE_MAX_RETRIES` (mínimo 1). Ex.: `SUPABASE_MAX_RETRIES=1`.

Testes:
- Incluímos testes unitários do cliente e teste de integração do `Trainer` com mocks de rede.
