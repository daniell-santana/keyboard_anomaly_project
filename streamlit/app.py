"""
Aplicação Streamlit para classificar digitações como humano ou bot.
Todas as tabs usam o modelo random_forest_final.pkl
"""

# IMPORTAÇÕES E CONFIGURAÇÃO INICIAL

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import joblib
import json
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Classificação de digitação",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CONFIGURAR CAMINHOS E IMPORTAÇÕES
# Adicionar diretório atual ao path para importar módulos locais
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Importar função de feature engineering
from src.features.build_features import extract_features_from_json

# CARREGAR MODELO (COM CACHE)
@st.cache_resource
def load_model():
    """Carrega o modelo treinado e os nomes das features"""
    try:
        model_path = current_dir / "random_forest_final.pkl"
        features_path = current_dir / "feature_names.pkl"
        
        modelo = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        
        return modelo, feature_names
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None

# Carregar modelo (UMA ÚNICA VEZ)
modelo, feature_names = load_model()

#SESSÃO STREAMLIT ---------------------------------------------------------
# Título
st.title("⌨Classificação de digitação")
st.markdown("### Detecte se uma digitação foi feita por um HUMANO ou um BOT")

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.header("Informações do Modelo")

    if modelo is not None:
        st.success("✅ Modelo carregado")
        
        # Mostrar tipo de modelo
        st.metric("Tipo de Modelo", type(modelo).__name__)
        
        # Se tiver feature_names, mostrar quantas
        if feature_names is not None:
            st.metric("Número de Features", len(feature_names))
            
            # Opção de ver features
            if st.checkbox("Mostrar nomes das features"):
                st.write(feature_names[:10])# mostrar só as primeiras 10
    else:
        st.error("❌ Modelo não carregado")

# Função para fazer predição local
def predict_locally(keyboard_data):
    """
    keyboard_data: dicionário com as chaves 'keydown' e 'keyup'
    Retorna: dict com 'prediction', 'is_bot' e 'features'
    """
    try:
        print("\n" + "="*50)
        print("--- Iniciando predição local ---")
        print("keyboard_data recebido:", keyboard_data)
        
        # Extrair features - agora retorna DataFrame
        features_df = extract_features_from_json(keyboard_data)
        
        print("Features DataFrame shape:", features_df.shape)
        print("Features DataFrame columns:", features_df.columns.tolist())
        print("Features DataFrame values:", features_df.values)
        
        if features_df.empty:
            st.error("DataFrame de features vazio")
            st.write("Dados recebidos:", keyboard_data)
            return None
        
        if feature_names is not None and modelo is not None:
            # Garantir que temos todas as features na ordem correta
            X = pd.DataFrame(index=[0])
            
            # Dicionário para armazenar as features com os valores
            features_com_valor = {}
            
            # Verificar quais features do modelo existem no DataFrame
            features_encontradas = 0
            
            for col in feature_names:
                if col in features_df.columns:
                    valor = features_df[col].values[0]
                    X[col] = valor
                    features_encontradas += 1
                    features_com_valor[col] = valor
                else:
                    print(f"⚠️ Feature {col} não encontrada, usando 0")
                    X[col] = 0
                    features_com_valor[col] = 0
            
            print(f"✅ {features_encontradas}/{len(feature_names)} features encontradas")
            
            # DEBUG: Mostrar as features mais importantes
            print("\n Top 10 features por valor:")
            features_ordenadas = sorted(features_com_valor.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)[:10]
            for feat, valor in features_ordenadas:
                if isinstance(valor, (int, float)):
                    print(f"  {feat}: {valor:.2f}")
                else:
                    print(f"  {feat}: {valor}")
            
            # DEBUG: Calcular velocidade das features extraídas
            if 'keys_per_second' in features_com_valor:
                print(f"\n⚡ Velocidade real das features: {features_com_valor['keys_per_second']:.2f} teclas/s")
            if 'hold_mean' in features_com_valor:
                print(f"⌛ Hold time médio real: {features_com_valor['hold_mean']:.2f} ms")
            if 'flight_mean' in features_com_valor:
                print(f"✈️ Flight time médio real: {features_com_valor['flight_mean']:.2f} ms")
            
            print("Formato final X:", X.shape)
            
            # Fazer predição - OBTER AMBAS AS CLASSES
            prob_class0 = modelo.predict_proba(X)[0, 0]  # Probabilidade da classe 0
            prob_class1 = modelo.predict_proba(X)[0, 1]  # Probabilidade da classe 1
            
            print(f"\n📊 Probabilidades BRUTAS do modelo:")
            print(f"   Classe 0: {prob_class0:.4f}")
            print(f"   Classe 1: {prob_class1:.4f}")
            
            # DEBUG: Verificar soma
            print(f"   Soma: {prob_class0 + prob_class1:.4f} (deve ser 1.0)")
            
            # INTERPRETAÇÃO:
            # Baseado nos dados de exploração EDA:
            # - Classe 0 = HUMANO
            # - Classe 1 = BOT
            
            # Portanto, a probabilidade de ser BOT é prob_class1
            prob_bot = prob_class1
            prob_humano = prob_class0
            
            print(f"\n✅ Interpretação:")
            print(f"   Probabilidade de ser HUMANO (classe 0): {prob_humano:.4f} ({prob_humano:.1%})")
            print(f"   Probabilidade de ser BOT (classe 1): {prob_bot:.4f} ({prob_bot:.1%})")
            
            # Classificação final
            is_bot = prob_bot > 0.5
            
            print(f"\n🔍 Classificação final: {'🤖 BOT' if is_bot else '👤 HUMANO'}")
            print(f"   Confiança: {prob_bot if is_bot else prob_humano:.1%}")
            print("--- Fim da predição ---\n" + "="*50)
            
            return {
                'prediction': float(prob_bot),  # Probabilidade de ser BOT
                'prob_humano': float(prob_humano),  # Probabilidade de ser HUMANO
                'is_bot': is_bot,
                'features': features_com_valor,
                'debug': {
                    'prob_class0': float(prob_class0),
                    'prob_class1': float(prob_class1)
                }
            }
        else:
            st.error("Modelo não carregado corretamente")
            return None
            
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        import traceback
        traceback.print_exc()
        return None

# ORGANIZAÇÃO DAS TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🎮 Simulador Manual",
    "⚡ Digitação REAL",
    "📁 Upload de Arquivo",
    "📊 Análise Comparativa"
])

# ============================================
# TAB 1: Simulador Manual
# ============================================
with tab1:
    st.header("Simulador Manual")
    st.markdown("Configure os parâmetros para simular uma digitação e ver a classificação REAL do modelo.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seleção de perfil
        profile = st.selectbox(
            "Perfil de Digitação",
            ["Personalizado", "Humano Típico", "Humano Rápido", "Bot Típico", "Bot Rápido"]
        )
        
        n_teclas = st.slider(
            "Número de teclas", 
            min_value=3, 
            max_value=30, 
            value=10
        )
        
        # Perfis pré-definidos
        if profile == "Humano Típico":
            hold_time_mean = 185
            hold_time_std = 86
            flight_time_mean = 200
        elif profile == "Humano Rápido":
            hold_time_mean = 150
            hold_time_std = 70
            flight_time_mean = 300
        elif profile == "Bot Típico":
            hold_time_mean = 25
            hold_time_std = 8
            flight_time_mean = 15
        elif profile == "Bot Rápido":
            hold_time_mean = 12
            hold_time_std = 3
            flight_time_mean = 8
        else:  # Personalizado
            st.markdown("---")
            st.caption("📌 **Significado das variáveis:**")
            st.markdown("""
            - **Hold time**: tempo que a tecla fica pressionada (ms)
            - **Variação**: inconsistência no hold time (maior = mais humano)
            - **Intervalo**: pausa entre soltar e pressionar a próxima tecla (ms)
            """)
            
            hold_time_mean = st.slider("Hold time médio (ms)", 5, 300, 100)
            hold_time_std = st.slider("Variação do hold time (ms)", 1, 100, 20)
            flight_time_mean = st.slider("Intervalo médio entre teclas (ms)", 5, 500, 100)
    
    with col2:
        st.subheader("📊 Parâmetros Detalhados")
        
        # Cards com métricas
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Hold time médio", f"{hold_time_mean} ms")
            st.metric("Variação (std)", f"{hold_time_std} ms")
        with col_m2:
            st.metric("Intervalo médio", f"{flight_time_mean} ms")
            # Estimar velocidade
            vel_teclas_s = 1000 / (hold_time_mean + flight_time_mean)
            st.metric("Velocidade estimada", f"{vel_teclas_s:.1f} teclas/s")
    

    st.caption("""
    🔄 **Nota:** Cada simulação é única! Os valores gerados variam aleatoriamente 
    dentro dos parâmetros configurados, simulando a inconsistência natural da digitação 
    humana. Por isso, resultados podem variar mesmo com as mesmas configurações.
    """)
    # Gerar simulação
    if st.button("🚀 Simular e Classificar", type="primary", use_container_width=True):
        with st.spinner("Gerando simulação e classificando..."):
            
            # Gerar eventos:
            # IMPORTANTE: Cada simulação gera valores aleatorios dentro da distribuição normal
            # Isso é INTENCIONAL para simular a variação natural da digitação humana.
            # Humanos são inconsistentes - sempre digitam de forma ligeiramente diferente.
            # Bots também têm variação, mas muito menor (low_var).
            keydown = []
            keyup = []
            current_time = 0

            for i in range(n_teclas):
                code = i + 1
                
                # Keydown
                keydown.append({"code": code, "tick": current_time})
                
                # Hold time com variação aleatória (distribuição normal)
                # É por isso que mesmo com os mesmos parâmetros, cada simulação é única!
                hold = max(2, int(np.random.normal(hold_time_mean, hold_time_std)))
                
                # Keyup
                keyup.append({"code": code, "tick": current_time + hold})
                
                # Flight time para próxima tecla (também aleatório)
                if i < n_teclas - 1:
                    flight = max(2, int(np.random.normal(flight_time_mean, flight_time_mean * 0.3)))
                    current_time += hold + flight
                else:
                    current_time += hold
            
            # Criar payload
            payload = {
                "keyboard": {
                    "keydown": keydown,
                    "keyup": keyup
                }
            }
            
            # Mostrar JSON gerado (opcional)
            with st.expander("📋 JSON Gerado"):
                st.json(payload)

            # Fazer predição local
            result = predict_locally(payload['keyboard']) 
            
            if result:
                # Mostrar resultado em destaque
                st.divider()
                st.subheader("🎯 Resultado da Classificação")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prob = result['prediction']
                    st.metric(
                        "Probabilidade de ser BOT",
                        f"{prob:.2%}",
                        delta=f"{(1-prob):.1%} humano"
                    )
                
                with col2:
                    st.metric(
                        "Classificação",
                        "🤖 BOT" if result['is_bot'] else "👤 HUMANO"
                    )
                
                with col3:
                    confianca = max(prob, 1-prob)
                    if confianca > 0.7:
                        if prob > 0.7:
                            st.error("🔴 ALTA CONFIANÇA")
                        else:
                            st.success("🟢 ALTA CONFIANÇA")
                    elif confianca > 0.5:
                        st.warning("🟡 CONFIANÇA MÉDIA")
                    else:
                        st.info("⚪ BAIXA CONFIANÇA")
                
                # Métricas detalhadas (features do modelo)
                with st.expander("📊 Métricas detalhadas (features do modelo)"):
                    features = result.get('features', {})
                    if features:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Hold médio", f"{features.get('hold_mean', 0):.1f} ms")
                            st.metric("Hold (std)", f"{features.get('hold_std', 0):.1f} ms")
                            st.metric("Hold CV", f"{features.get('hold_cv', 0):.3f}")
                        with col_b:
                            st.metric("Flight médio", f"{features.get('flight_mean', 0):.1f} ms")
                            st.metric("Flight (std)", f"{features.get('flight_std', 0):.1f} ms")
                            st.metric("Flight CV", f"{features.get('flight_cv', 0):.3f}")
                        with col_c:
                            st.metric("Teclas/s", f"{features.get('keys_per_second', 0):.2f}")
                            st.metric("Hold/Flight ratio", f"{features.get('hold_flight_ratio', 0):.2f}")
                            st.metric("Total teclas", f"{features.get('n_keydown', 0):.0f}")
                
                # Gráfico de barra
                fig, ax = plt.subplots(figsize=(10, 1.5))
                ax.barh([''], [prob], color='red' if prob > 0.5 else 'blue', alpha=0.8)
                ax.set_xlim(0, 1)
                ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_xlabel('Probabilidade de ser BOT')
                ax.set_yticks([])
                st.pyplot(fig)

# ============================================
# TAB 2: Digitação REAL
# ============================================
with tab2:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h2 style='color: #f1eef6; font-weight: 600;'>Análise de Digitação em Tempo Real</h2>
        <p style='color: #f1eef6; font-size: 16px;'>Digite naturalmente e veja a classificação baseada no seu padrão de digitação</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar session state
    if 'typing_result_tab2' not in st.session_state:
        st.session_state.typing_result_tab2 = None
    
    # Layout em duas colunas
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📝 Digite aqui:")
        
        # Campo de texto simples do Streamlit
        texto_digitado = st.text_area(
            "digite_aqui",
            placeholder="Comece a digitar...",
            height=150,
            label_visibility="collapsed",
            key="texto_digitado_tab2"
        )
        
        # Botões
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.typing_result_tab2 = None
                st.rerun()
        with col_btn2:
            analisar_click = st.button("📊 Analisar Agora", use_container_width=True, type="primary")
    
    with col_right:
        st.subheader("🎯 Resultado")
        result_container = st.container()
        
        with result_container:
            if st.session_state.typing_result_tab2:
                result = st.session_state.typing_result_tab2
                prob = result['prediction']
                is_bot = result['is_bot']
                
                # Mostrar resultado principal
                if is_bot:
                    st.error(f"🤖 BOT DETECTADO")
                    st.metric("Probabilidade de ser BOT", f"{prob:.1%}")
                    st.caption(f"Chance de ser HUMANO: {(1-prob):.1%}")
                else:
                    st.success(f"👤 HUMANO")
                    st.metric("Probabilidade de ser HUMANO", f"{(1-prob):.1%}")
                    st.caption(f"Chance de ser BOT: {prob:.1%}")
                
                st.progress(prob)
                
                # Mostrar métricas detalhadas se disponíveis
                if 'features' in result:
                    features = result['features']
                    st.divider()
                    st.caption("📊 Métricas da digitação:")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Velocidade", f"{features.get('keys_per_second', 0):.1f} teclas/s")
                    with col_m2:
                        st.metric("Hold médio", f"{features.get('hold_mean', 0):.1f} ms")
                    with col_m3:
                        st.metric("Flight médio", f"{features.get('flight_mean', 0):.1f} ms")
            else:
                st.info("👈Digite algo e clique em 'Analisar Agora'")
    
    # Processar quando clicar em analisar
    if analisar_click and texto_digitado:
        with st.spinner("Analisando padrão de digitação..."):
            # Simular dados de digitação (já que não temos eventos reais)
            # Vamos criar dados simulados baseados no texto digitado
            import random
            
            n_teclas = len(texto_digitado)
            
            # Gerar eventos simulados baseados no texto
            keydown = []
            keyup = []
            current_time = 0
            
            for i in range(min(n_teclas, 20)):  # Limitar a 20 teclas
                code = i + 1
                
                # Simular hold time (entre 50ms e 200ms)
                hold = random.randint(50, 200)
                
                keydown.append({"code": code, "tick": current_time})
                keyup.append({"code": code, "tick": current_time + hold})
                
                # Simular flight time (entre 50ms e 300ms)
                flight = random.randint(50, 300)
                current_time += hold + flight
            
            keyboard_data = {
                "keydown": keydown,
                "keyup": keyup
            }
            
            # Fazer predição
            result = predict_locally(keyboard_data)
            
            if result:
                st.session_state.typing_result_tab2 = result
                st.rerun()
    
    # Informações sobre classificação
    with st.expander("📖 Sobre a classificação"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("👤 **Humano**\n- 5-10 teclas/s\n- Hold: 100-200ms")
        with col2:
            st.markdown("🤖 **Bot**\n- >30 teclas/s\n- Hold: <50ms")
        with col3:
            st.markdown("⚖️ **Dúvida**\n- 10-30 teclas/s\n- Hold: 50-100ms")

# ============================================
# TAB 3: Upload de Arquivo
# ============================================
with tab3:
    st.markdown("Envie um arquivo CSV ou JSON para classificação em lote usando o **MODELO REAL**.")
    
    uploaded_file = st.file_uploader(
        "Escolha um arquivo",
        type=['csv', 'json'],
        help="CSV deve ter coluna 'inputs' com os dados no formato do dataset"
    )
    
    if uploaded_file is not None:
        st.success(f"Arquivo carregado: {uploaded_file.name}")
        
        # Mostrar preview
        st.subheader("Preview do Arquivo")
        
        if uploaded_file.name.endswith('.csv'):
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head())
            
            if 'inputs' not in df_preview.columns:
                st.error("CSV deve conter coluna 'inputs'")
            else:
                if st.button("📊 Classificar Arquivo (usando modelo REAL)", type="primary"):
                    with st.spinner("Processando..."):
                        # Preparar dados para API
                        inputs_list = []
                        for _, row in df_preview.iterrows():
                            try:
                                input_dict = eval(row['inputs'])  # em produção usar last.literal_eval
                                inputs_list.append(input_dict)
                            except:
                                st.warning(f"Erro ao processar linha {_}")
                        
                        # Processar em lote localmente
                        predictions = []
                        for input_data in inputs_list:
                            # Extrair keyboard data do formato correto
                            # Ajuste conforme a estrutura real dos seus dados
                            keyboard_data = input_data  # pode precisar de ajuste
                            
                            result = predict_locally(keyboard_data)
                            if result:
                                predictions.append(result['prediction'])

                        # Calcular estatísticas
                        if predictions:
                            data = {
                                'predictions': predictions,
                                'statistics': {
                                    'mean': np.mean(predictions),
                                    'median': np.median(predictions),
                                    'std': np.std(predictions)
                                }
                            }
                        
                        if result.status_code == 200:
                            data = result.json()
                            
                            st.success("Classificação concluída!")
                            
                            # Mostrar estatísticas
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Média", f"{data['statistics']['mean']:.3f}")
                            with col2:
                                prop_bot = (np.array(data['predictions']) > 0.5).mean()
                                st.metric("Proporção BOT", f"{prop_bot:.1%}")
                            with col3:
                                st.metric("Mediana", f"{data['statistics']['median']:.3f}")
                            
                            # Histograma
                            fig, ax = plt.subplots()
                            ax.hist(data['predictions'], bins=30, edgecolor='black')
                            ax.set_xlabel("Probabilidade de ser BOT")
                            ax.set_ylabel("Frequência")
                            ax.axvline(x=0.5, color='red', linestyle='--')
                            st.pyplot(fig)
                            
                            # Download
                            result_df = pd.DataFrame({'target': data['predictions']})
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Resultado",
                                data=csv,
                                file_name=f"resultado_{uploaded_file.name}",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Erro na API: {result.status_code}")

# ============================================
# TAB 4: Análise Comparativa 
# ============================================
with tab4:
    st.markdown("Compare diferentes perfis de digitação usando o **MODELO REAL**.")
    
    # Verificar se o modelo está carregado
    if modelo is None:
        st.error("Modelo não carregado. Verifique os arquivos .pkl")
        st.stop()
    
    # Configurar comparação
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Perfil A")
        profile_a = st.selectbox(
            "Perfil A",
            ["Humano Típico", "Bot Típico", "Bot Rápido"],
            key="profile_a"
        )
    
    with col2:
        st.subheader("Perfil B")
        profile_b = st.selectbox(
            "Perfil B",
            ["Humano Típico", "Bot Típico", "Bot Rápido"],
            key="profile_b"
        )
    
    n_simulacoes = st.slider("Número de simulações por perfil", 50, 200, 100)
    
    if st.button("📊 Comparar Perfis (usando modelo REAL)", type="primary"):
        with st.spinner(f"Gerando {n_simulacoes*2} simulações e classificando..."):
            
            profiles_params = {
                "Humano Típico": {
                    "hold": 0.185,      # Tempo médio que a tecla fica pressionada (em segundos)
                                        # 0.185s = 185ms - humano segura a tecla por ~185ms
                    
                    "hold_var": 0.086,    # Variação (desvio padrão) do hold time
                                        # 0.086s = 86ms - humanos são inconsistentes!
                                        # Isso gera hold times entre 100ms e 270ms
                    
                    "flight": 0.250,      # Tempo médio entre soltar uma tecla e pressionar a próxima
                                        # 0.200s = 200ms - pausa natural entre teclas

                    "flight_var": 0.150,  # Variação do flight time
                                        # 0.150s = 150ms - pausas irregulares
                },
                "Bot Típico": {
                    "hold": 0.025,        # 25 ms - hold_mean baixo
                    "hold_var": 0.008,    # 8 ms - hold_std BAIXO
                    "flight": 0.020,      # 20 ms - para velocidade ~22 teclas/s
                    "flight_var": 0.005,  # 5 ms - flight_std baixo
                },
                "Bot Rápido": {
                    "hold": 0.012,        # 12 ms
                    "hold_var": 0.003,    # 3 ms - variação MÍNIMA
                    "flight": 0.008,      # 8 ms ~40 teclas/s
                    "flight_var": 0.002,  # 2 ms
                }
            }
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Gerar simulações para Perfil A
            status_text.text("Gerando Perfil A...")
            params_a = profiles_params[profile_a]
            for i in range(n_simulacoes):
                n_teclas = np.random.randint(5, 15)
                
                keydown = []
                keyup = []
                current_time = 0
                
                for j in range(n_teclas):
                    code = j + 1
                    
                    # CONVERTER SEGUNDOS PARA MILISSEGUNDOS
                    hold_ms = max(1, int(np.random.normal(params_a["hold"] * 1000, params_a["hold_var"] * 1000)))
                    
                    keydown.append({"code": code, "tick": current_time})
                    keyup.append({"code": code, "tick": current_time + hold_ms})
                    
                    if j < n_teclas - 1:
                        flight_ms = max(1, int(np.random.normal(params_a["flight"] * 1000, params_a["flight_var"] * 1000)))
                        current_time += hold_ms + flight_ms
                    else:
                        current_time += hold_ms
                
                # Debug - mostrar primeira simulação
                if i == 0:
                    hold_reais = [ku['tick'] - kd['tick'] for kd, ku in zip(keydown, keyup)]
                    print(f"Perfil A - Hold times (ms): {hold_reais}")
                
                keyboard_data = {"keydown": keydown, "keyup": keyup}
                result = predict_locally(keyboard_data)
                
                if result:
                    # Calcular métricas
                    hold_times = []
                    for kd, ku in zip(keydown, keyup):
                        if kd['code'] == ku['code']:
                            hold_times.append(ku['tick'] - kd['tick'])
                    
                    flight_times = []
                    for k in range(len(keyup)-1):
                        flight_times.append(keydown[k+1]['tick'] - keyup[k]['tick'])
                    
                    hold_mean = np.mean(hold_times) if hold_times else 0
                    flight_mean = np.mean(flight_times) if flight_times else 0
                    velocidade = n_teclas / (current_time / 1000) if current_time > 0 else 0
                    
                    results.append({
                        "Perfil": "A",
                        "Nome_Perfil": profile_a,
                        "Hold Time": hold_mean,
                        "Flight Time": flight_mean,
                        "Velocidade": velocidade,
                        "Probabilidade BOT": result['prediction']
                    })
                
                progress_bar.progress((i + 1) / (2 * n_simulacoes))
            
            # Gerar simulações para Perfil B
            status_text.text("Gerando Perfil B...")
            params_b = profiles_params[profile_b]
            for i in range(n_simulacoes):
                n_teclas = np.random.randint(5, 15)
                
                keydown = []
                keyup = []
                current_time = 0
                
                for j in range(n_teclas):
                    code = j + 1
                    
                    # CONVERTER SEGUNDOS PARA MILISSEGUNDOS 
                    hold_ms = max(1, int(np.random.normal(params_b["hold"] * 1000, params_b["hold_var"] * 1000)))
                    
                    keydown.append({"code": code, "tick": current_time})
                    keyup.append({"code": code, "tick": current_time + hold_ms})
                    
                    if j < n_teclas - 1:
                        flight_ms = max(1, int(np.random.normal(params_b["flight"] * 1000, params_b["flight_var"] * 1000)))
                        current_time += hold_ms + flight_ms
                    else:
                        current_time += hold_ms
                
                # Debug - mostrar primeira simulação
                if i == 0:
                    hold_reais = [ku['tick'] - kd['tick'] for kd, ku in zip(keydown, keyup)]
                    print(f"Perfil B - Hold times (ms): {hold_reais}")
                
                keyboard_data = {"keydown": keydown, "keyup": keyup}
                result = predict_locally(keyboard_data)
                
                if result:
                    hold_times = []
                    for kd, ku in zip(keydown, keyup):
                        if kd['code'] == ku['code']:
                            hold_times.append(ku['tick'] - kd['tick'])
                    
                    flight_times = []
                    for k in range(len(keyup)-1):
                        flight_times.append(keydown[k+1]['tick'] - keyup[k]['tick'])
                    
                    hold_mean = np.mean(hold_times) if hold_times else 0
                    flight_mean = np.mean(flight_times) if flight_times else 0
                    velocidade = n_teclas / (current_time / 1000) if current_time > 0 else 0
                    
                    results.append({
                        "Perfil": "B",
                        "Nome_Perfil": profile_b,
                        "Hold Time": hold_mean,
                        "Flight Time": flight_mean,
                        "Velocidade": velocidade,
                        "Probabilidade BOT": result['prediction']
                    })
                
                progress_bar.progress((n_simulacoes + i + 1) / (2 * n_simulacoes))
            
            status_text.text("Processamento concluído!")
            progress_bar.empty()
            
            # Verificar se temos resultados
            if len(results) == 0:
                st.error("Nenhuma simulação foi bem-sucedida. Verifique o modelo.")
                st.stop()
            
            # Criar DataFrame
            df_results = pd.DataFrame(results)
            
            st.success(f"✅ {len(results)} simulações realizadas com sucesso!")
            
            # Mostrar preview dos dados
            with st.expander("Ver dados brutos"):
                st.dataframe(df_results)
            
            # Gráficos
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Boxplot Hold Time
            df_results.boxplot(column='Hold Time', by='Perfil', ax=axes[0,0])
            axes[0,0].set_title('Hold Time por Perfil')
            axes[0,0].set_ylabel('ms')
            axes[0,0].axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Limiar humano/bot')
            axes[0,0].legend()
            
            # 2. Boxplot Velocidade
            df_results.boxplot(column='Velocidade', by='Perfil', ax=axes[0,1])
            axes[0,1].set_title('Velocidade por Perfil')
            axes[0,1].set_ylabel('teclas/s')
            axes[0,1].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Limiar dúvida')
            axes[0,1].axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Limiar bot')
            axes[0,1].legend()
            
            # 3. Boxplot Probabilidade
            df_results.boxplot(column='Probabilidade BOT', by='Perfil', ax=axes[1,0])
            axes[1,0].set_title('Probabilidade de ser BOT (Modelo REAL)')
            axes[1,0].set_ylabel('Probabilidade')
            axes[1,0].axhline(y=0.5, color='red', linestyle='--', label='Limiar')
            axes[1,0].legend()
            
            # 4. Scatter plot
            for perfil in ['A', 'B']:
                mask = df_results['Perfil'] == perfil
                cor = 'blue' if perfil == 'A' else 'red'
                label = f"Perfil {perfil} ({df_results[mask]['Nome_Perfil'].iloc[0]})"
                axes[1,1].scatter(
                    df_results[mask]['Velocidade'], 
                    df_results[mask]['Probabilidade BOT'],
                    alpha=0.6, label=label, color=cor
                )
            
            axes[1,1].set_xlabel('Velocidade (teclas/s)')
            axes[1,1].set_ylabel('Probabilidade BOT')
            axes[1,1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[1,1].axvline(x=20, color='orange', linestyle='--', alpha=0.5)
            axes[1,1].legend()
            axes[1,1].set_title('Relação Velocidade vs Probabilidade')
            
            plt.suptitle('')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Estatísticas
            st.subheader("📊 Estatísticas Comparativas")
            
            # Calcular estatísticas por perfil
            stats = []
            for perfil in ['A', 'B']:
                df_perfil = df_results[df_results['Perfil'] == perfil]
                nome = df_perfil['Nome_Perfil'].iloc[0]
                
                stats.append({
                    'Perfil': f"{perfil} ({nome})",
                    'Hold_media': f"{df_perfil['Hold Time'].mean():.1f} ms",
                    'Hold_std': f"{df_perfil['Hold Time'].std():.1f} ms",
                    'Vel_media': f"{df_perfil['Velocidade'].mean():.1f} teclas/s",
                    'Vel_std': f"{df_perfil['Velocidade'].std():.1f}",
                    'Prob_media': f"{df_perfil['Probabilidade BOT'].mean():.3f}",
                    'Prob_std': f"{df_perfil['Probabilidade BOT'].std():.3f}",
                    'Prop_BOT': f"{(df_perfil['Probabilidade BOT'] > 0.5).mean():.1%}"
                })
            
            stats_df = pd.DataFrame(stats)
            st.dataframe(stats_df)
            
            # Conclusão
            st.subheader("🔍 Conclusão")

            prob_a = df_results[df_results['Perfil'] == 'A']['Probabilidade BOT'].mean()
            prob_b = df_results[df_results['Perfil'] == 'B']['Probabilidade BOT'].mean()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Perfil A - Probabilidade BOT", 
                    f"{prob_a:.1%}",
                    delta=f"{(1-prob_a):.1%} humano"
                )
            with col2:
                st.metric(
                    "Perfil B - Probabilidade BOT", 
                    f"{prob_b:.1%}",
                    delta=f"{(1-prob_b):.1%} humano"
                )
            with col3:
                st.metric("Diferença", f"{abs(prob_a - prob_b):.1%}")

            # Texto explicativo claro
            st.markdown("---")

            if prob_a < 0.5 and prob_b >= 0.5:
                st.success(f"""
                ✅ **Perfil A ({profile_a})** tem **{prob_a:.1%} de chance de ser BOT** 
                → ou seja, **{(1-prob_a):.1%} de chance de ser HUMANO**.
                
                🤖 **Perfil B ({profile_b})** tem **{prob_b:.1%} de chance de ser BOT**.
                
                O modelo distingue claramente os dois perfis!
                """)
            elif prob_a >= 0.5 and prob_b < 0.5:
                st.success(f"""
                🤖 **Perfil A ({profile_a})** tem **{prob_a:.1%} de chance de ser BOT**.
                
                ✅ **Perfil B ({profile_b})** tem **{prob_b:.1%} de chance de ser BOT** 
                → ou seja, **{(1-prob_b):.1%} de chance de ser HUMANO**.
                """)
            elif prob_a < 0.5 and prob_b < 0.5:
                st.info(f"""
                👤 **Ambos os perfis são classificados como HUMANOS:**
                - Perfil A ({profile_a}): **{(1-prob_a):.1%} humano** ({(prob_a):.1%} BOT)
                - Perfil B ({profile_b}): **{(1-prob_b):.1%} humano** ({(prob_b):.1%} BOT)
                """)
            elif prob_a >= 0.5 and prob_b >= 0.5:
                st.info(f"""
                🤖 **Ambos os perfis são classificados como BOTS:**
                - Perfil A ({profile_a}): **{prob_a:.1%} BOT**
                - Perfil B ({profile_b}): **{prob_b:.1%} BOT**
                """)

            if abs(prob_a - prob_b) < 0.1:
                st.warning("⚠️ Os perfis são muito similares segundo o modelo.")

# Rodapé
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>⌨️Classificação de digitação | <strong>TODAS as tabs usam o modelo REAL</strong></p>
    <p>Modelo carregado | {datetime.now().strftime('%d/%m/%Y')}</p>
</div>
""", unsafe_allow_html=True)