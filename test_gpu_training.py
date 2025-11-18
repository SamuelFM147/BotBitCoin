#!/usr/bin/env python3
"""Teste rápido de treinamento com GPU"""

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def test_gpu_training():
    print("=== Teste de Treinamento com GPU ===")
    print(f"GPU disponível: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    if torch.cuda.is_available():
        print("\n🚀 Iniciando teste de treinamento com GPU...")
        
        # Criar ambiente simples
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        
        # Criar agente PPO
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",  # Usar GPU
            n_steps=128,    # Passos menores para teste rápido
            batch_size=64
        )
        
        # Treinar por poucos passos
        print("Treinando por 1000 timesteps...")
        model.learn(total_timesteps=1000)
        
        print("✅ Treinamento com GPU concluído com sucesso!")
        
        # Testar inferência
        obs = env.reset()
        action, _states = model.predict(obs)
        print(f"✅ Inferência funcionando! Ação: {action}")
        
        env.close()
    else:
        print("❌ GPU não disponível")

if __name__ == "__main__":
    test_gpu_training()