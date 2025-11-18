#!/usr/bin/env python3
"""Teste de disponibilidade do CUDA"""

import torch

def test_cuda():
    print("=== Teste de CUDA ===")
    print(f"PyTorch versão: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA versão: {torch.version.cuda}")
        
        # Testar memória
        props = torch.cuda.get_device_properties(0)
        print(f"Memória total: {props.total_memory // 1024**3} GB")
        print(f"Multi-processadores: {props.multi_processor_count}")
        
        # Testar operação simples na GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = torch.tensor([4.0, 5.0, 6.0]).cuda()
            z = x + y
            print(f"Teste GPU: {x} + {y} = {z}")
            print("✅ GPU funcionando corretamente!")
        except Exception as e:
            print(f"❌ Erro ao usar GPU: {e}")
    else:
        print("❌ CUDA não disponível - usando CPU")

if __name__ == "__main__":
    test_cuda()