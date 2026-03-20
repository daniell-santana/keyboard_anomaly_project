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
    # Se for string, converter para dict
    if isinstance(data, str):
        data = json.loads(data)
    
    features = []
    
    # Garantir que é uma lista
    inputs_list = data if isinstance(data, list) else [data]
    
    for item in inputs_list:
        try:
            keyboard_data = item.get('keyboard', {})
            keydown = keyboard_data.get('keydown', [])
            keyup = keyboard_data.get('keyup', [])
            
            feat = {}
            
            # Features básicas
            feat['length'] = item.get('length', len(keydown))
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
                keydown_queue = defaultdict(list)
                for event in sorted(keydown, key=lambda x: x['tick']):
                    keydown_queue[event['code']].append(event['tick'])
                for event in sorted(keyup, key=lambda x: x['tick']):
                    code = event['code']
                    if keydown_queue[code]:
                        press_time = keydown_queue[code].pop(0)
                        hold_time = event['tick'] - press_time
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
            
            # Flight times
            all_events = []
            for e in keydown:
                all_events.append(('down', e['code'], e['tick']))
            for e in keyup:
                all_events.append(('up', e['code'], e['tick']))
            
            flight_times = []
            if len(all_events) > 1:
                all_events.sort(key=lambda x: x[2])
                for i in range(1, len(all_events)):
                    flight_times.append(all_events[i][2] - all_events[i-1][2])
            
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
                    press_press.append(keydown_times[i] - keydown_times[i-1])
            
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
            
        except Exception as e:
            print(f"Erro ao processar: {e}")
            # Fallback: features zero
            feat = {k: 0 for k in ['length', 'n_keydown', 'n_keyup', 'n_unique_codes',
                                   'total_time', 'first_tick', 'hold_mean', 'hold_std',
                                   'hold_min', 'hold_max', 'hold_median', 'hold_q25',
                                   'hold_q75', 'hold_cv', 'flight_mean', 'flight_std',
                                   'flight_min', 'flight_max', 'flight_median', 'flight_cv',
                                   'press_press_mean', 'press_press_std', 'press_press_cv',
                                   'keys_per_second', 'hold_flight_ratio']}
            features.append(feat)
    
    return pd.DataFrame(features)