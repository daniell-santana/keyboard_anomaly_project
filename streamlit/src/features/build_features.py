"""
Funções de feature engineering compartilhadas entre API e Streamlit.
Cópia das funções desenvolvidas nos notebooks.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import ast
import json
from typing import Dict, List, Any, Union

def parse_inputs(inputs_series):
    """Converte strings de inputs para dicionários."""
    if isinstance(inputs_series, str):
        return ast.literal_eval(inputs_series)
    return inputs_series.apply(ast.literal_eval)

def extract_features_from_json(data: Union[Dict, str]) -> pd.DataFrame:
    """
    Extrai features a partir do JSON de entrada.
    Versão robusta para dados incompletos.
    """
    print("="*50)
    print("Dados recebidos:", data)
    print("Tipo:", type(data))
    
    # Se for string, converter para dict
    if isinstance(data, str):
        try:
            data = json.loads(data)
            print("Convertido de string para dict")
        except json.JSONDecodeError as e:
            print(f"Erro ao converter string JSON: {e}")
            return pd.DataFrame()
    
    features = []
    
    # Garantir que é uma lista
    if isinstance(data, dict):
        # Se já tem a estrutura esperada, usar diretamente
        if 'keydown' in data and 'keyup' in data:
            # Já é o keyboard_data
            inputs_list = [{'keyboard': data}] 
        elif 'keyboard' in data:
            # Tem a chave keyboard
            inputs_list = [data]
        else:
            # Formato desconhecido, tentar usar como está
            inputs_list = [{'keyboard': data}]
    elif isinstance(data, list):
        inputs_list = data
    else:
        print(f"Tipo não esperado: {type(data)}")
        return pd.DataFrame()
    
    print(f"Processando {len(inputs_list)} item(s)")
    
    for idx, item in enumerate(inputs_list):
        try:
            print(f"Item {idx}: {item}")
            
            # Extrair keyboard_data
            if isinstance(item, dict):
                if 'keyboard' in item:
                    keyboard_data = item.get('keyboard', {})
                else:
                    # Se não tem 'keyboard', assumir que o próprio item é o keyboard_data
                    keyboard_data = item
            else:
                keyboard_data = {}
            
            keydown = keyboard_data.get('keydown', [])
            keyup = keyboard_data.get('keyup', [])
            
            print(f"keydown: {len(keydown)} eventos, keyup: {len(keyup)} eventos")
            
            feat = {}
            
            # Features básicas
            feat['length'] = len(keydown)  # Simplificado
            feat['n_keydown'] = len(keydown)
            feat['n_keyup'] = len(keyup)
            feat['n_unique_codes'] = len(set([e['code'] for e in keydown])) if keydown else 0
            
            # Tempos totais
            if keydown:
                max_keydown = max(e['tick'] for e in keydown)
                min_keydown = min(e['tick'] for e in keydown)
            else:
                max_keydown = 0
                min_keydown = 0
                
            if keyup:
                max_keyup = max(e['tick'] for e in keyup)
            else:
                max_keyup = 0
            
            feat['total_time'] = max(max_keyup, max_keydown)
            feat['first_tick'] = min_keydown if keydown else 0
            
            # Hold times
            hold_times = []
            if keydown and keyup:
                # Criar dicionário de keydown por código
                keydown_dict = {}
                for event in keydown:
                    code = event['code']
                    if code not in keydown_dict:
                        keydown_dict[code] = []
                    keydown_dict[code].append(event['tick'])
                
                # Processar keyup
                for event in keyup:
                    code = event['code']
                    if code in keydown_dict and keydown_dict[code]:
                        press_time = keydown_dict[code].pop(0)
                        hold_time = event['tick'] - press_time
                        if hold_time > 0:  # Apenas hold times positivos
                            hold_times.append(hold_time)
            
            if hold_times:
                feat['hold_mean'] = np.mean(hold_times)
                feat['hold_std'] = np.std(hold_times)
                feat['hold_min'] = np.min(hold_times)
                feat['hold_max'] = np.max(hold_times)
                feat['hold_median'] = np.median(hold_times)
                feat['hold_q25'] = np.percentile(hold_times, 25)
                feat['hold_q75'] = np.percentile(hold_times, 75)
                feat['hold_cv'] = feat['hold_std'] / feat['hold_mean'] if feat['hold_mean'] > 0 else 0
            else:
                feat['hold_mean'] = feat['hold_std'] = 0
                feat['hold_min'] = feat['hold_max'] = 0
                feat['hold_median'] = feat['hold_q25'] = feat['hold_q75'] = 0
                feat['hold_cv'] = 0
            
            # Flight times (intervalos entre eventos)
            all_events = []
            for e in keydown:
                all_events.append(('down', e['code'], e['tick']))
            for e in keyup:
                all_events.append(('up', e['code'], e['tick']))
            
            flight_times = []
            if len(all_events) > 1:
                all_events.sort(key=lambda x: x[2])
                for i in range(1, len(all_events)):
                    diff = all_events[i][2] - all_events[i-1][2]
                    if diff > 0:
                        flight_times.append(diff)
            
            if flight_times:
                feat['flight_mean'] = np.mean(flight_times)
                feat['flight_std'] = np.std(flight_times)
                feat['flight_min'] = np.min(flight_times)
                feat['flight_max'] = np.max(flight_times)
                feat['flight_median'] = np.median(flight_times)
                feat['flight_cv'] = feat['flight_std'] / feat['flight_mean'] if feat['flight_mean'] > 0 else 0
            else:
                feat['flight_mean'] = feat['flight_std'] = 0
                feat['flight_min'] = feat['flight_max'] = 0
                feat['flight_median'] = feat['flight_cv'] = 0
            
            # Press-press intervals
            press_press = []
            if len(keydown) > 1:
                keydown_times = [e['tick'] for e in sorted(keydown, key=lambda x: x['tick'])]
                for i in range(1, len(keydown_times)):
                    diff = keydown_times[i] - keydown_times[i-1]
                    if diff > 0:
                        press_press.append(diff)
            
            if press_press:
                feat['press_press_mean'] = np.mean(press_press)
                feat['press_press_std'] = np.std(press_press)
                feat['press_press_cv'] = feat['press_press_std'] / feat['press_press_mean'] if feat['press_press_mean'] > 0 else 0
            else:
                feat['press_press_mean'] = 0
                feat['press_press_std'] = 0
                feat['press_press_cv'] = 0
            
            # Velocidade
            if feat['total_time'] > 0:
                feat['keys_per_second'] = feat['n_keydown'] / (feat['total_time'] / 1000)
            else:
                feat['keys_per_second'] = 0
            
            # Ratio
            if feat['flight_mean'] > 0:
                feat['hold_flight_ratio'] = feat['hold_mean'] / feat['flight_mean']
            else:
                feat['hold_flight_ratio'] = 0
            
            features.append(feat)
            print(f"Features geradas para item {idx}: {len(feat)} features")
            
        except Exception as e:
            print(f"Erro ao processar item {idx}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: features zero
            feat = {k: 0 for k in ['length', 'n_keydown', 'n_keyup', 'n_unique_codes',
                                   'total_time', 'first_tick', 'hold_mean', 'hold_std',
                                   'hold_min', 'hold_max', 'hold_median', 'hold_q25',
                                   'hold_q75', 'hold_cv', 'flight_mean', 'flight_std',
                                   'flight_min', 'flight_max', 'flight_median', 'flight_cv',
                                   'press_press_mean', 'press_press_std', 'press_press_cv',
                                   'keys_per_second', 'hold_flight_ratio']}
            features.append(feat)
    
    print(f"Total de features geradas: {len(features)}")
    if features:
        print("Primeiro item features:", features[0])
    print("="*50)
    
    if not features:
        # Se não gerou nenhuma feature, retornar DataFrame vazio
        return pd.DataFrame()
    
    return pd.DataFrame(features)