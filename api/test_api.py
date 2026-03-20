"""
Script para testar a API localmente.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Testa endpoint /health"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_predict_single():
    """Testa predição única"""
    
    # Exemplo de payload (baseado nos dados reais)
    payload = {
        "keyboard": {
            "keydown": [
                {"code": 467, "tick": 0},
                {"code": 636, "tick": 87}
            ],
            "keyup": [
                {"code": 467, "tick": 98},
                {"code": 636, "tick": 284}
            ]
        },
        "length": 2
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nPredict single - Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predição: {result['prediction']:.4f}")
        print(f"É bot? {result['is_bot']}")
        return result
    else:
        print(f"Erro: {response.text}")
        return None

def test_predict_batch():
    """Testa predição em lote"""
    
    payload = {
        "inputs": [
            {
                "keyboard": {
                    "keydown": [{"code": 467, "tick": 0}],
                    "keyup": [{"code": 467, "tick": 98}]
                },
                "length": 1
            },
            {
                "keyboard": {
                    "keydown": [{"code": 1, "tick": 0}, {"code": 2, "tick": 17}],
                    "keyup": [{"code": 1, "tick": 16}, {"code": 2, "tick": 32}]
                },
                "length": 2
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=payload)
    print(f"\nPredict batch - Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predições: {result['predictions']}")
        print(f"Estatísticas: {result['statistics']}")
        return result
    else:
        print(f"Erro: {response.text}")
        return None

def test_feature_importance():
    """Testa endpoint de importância das features"""
    
    response = requests.get(f"{BASE_URL}/feature_importance")
    print(f"\nFeature importance - Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Top 5 features:")
        for i, feat in enumerate(result['feature_importance'][:5]):
            print(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}")
        return result
    else:
        print(f"Erro: {response.text}")
        return None

def test_simulate_bot():
    """Testa simulação de bot (velocidade alta)"""
    
    # Simular bot: hold times muito curtos (10-20ms)
    payload = {
        "keyboard": {
            "keydown": [
                {"code": 1, "tick": 0},
                {"code": 2, "tick": 15},
                {"code": 3, "tick": 30},
                {"code": 4, "tick": 45}
            ],
            "keyup": [
                {"code": 1, "tick": 12},
                {"code": 2, "tick": 27},
                {"code": 3, "tick": 42},
                {"code": 4, "tick": 57}
            ]
        },
        "length": 4
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nSimulação de BOT - Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Hold times: 12ms")
        print(f"Predição: {result['prediction']:.4f}")
        print(f"Classificado como: {'BOT' if result['is_bot'] else 'HUMANO'}")
        return result
    else:
        print(f"Erro: {response.text}")
        return None

def test_simulate_human():
    """Testa simulação de humano (velocidade normal)"""
    
    # Simular humano: hold times normais (100-150ms)
    payload = {
        "keyboard": {
            "keydown": [
                {"code": 1, "tick": 0},
                {"code": 2, "tick": 200},
                {"code": 3, "tick": 380},
                {"code": 4, "tick": 590}
            ],
            "keyup": [
                {"code": 1, "tick": 120},
                {"code": 2, "tick": 320},
                {"code": 3, "tick": 500},
                {"code": 4, "tick": 720}
            ]
        },
        "length": 4
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nSimulação de HUMANO - Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Hold times: ~120ms")
        print(f"Predição: {result['prediction']:.4f}")
        print(f"Classificado como: {'BOT' if result['is_bot'] else 'HUMANO'}")
        return result
    else:
        print(f"Erro: {response.text}")
        return None

if __name__ == "__main__":
    print("="*50)
    print("TESTANDO API DE CLASSIFICAÇÃO")
    print("="*50)
    
    # Aguardar API iniciar
    print("\nAguardando API iniciar...")
    time.sleep(2)
    
    # Testes
    test_health()
    test_predict_single()
    test_predict_batch()
    test_feature_importance()
    test_simulate_bot()
    test_simulate_human()
    
    print("\n" + "="*50)
    print("TESTES CONCLUÍDOS!")
    print("="*50)