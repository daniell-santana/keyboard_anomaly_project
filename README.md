## 🚀 Tech Stack

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Image-0db7ed?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/daniellsantanaa/keyboard-streamlit)

[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-228B22?style=for-the-badge)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-blue?style=for-the-badge)]()

# Keyboard Typing Classification

## 🎯 Sobre o Desafio

Este projeto tem como objetivo classificar padrões de digitação a partir da dinâmica temporal de eventos keydown e keyup. Utilizando técnicas de Machine Learning, o modelo é capaz de identificar e diferenciar comportamentos de digitação humana e automatizada (bots), com foco em detecção de anomalias e segurança comportamental.

------------------------------------------------------------------------

## 📁 Estrutura do Projeto
```
├── README.md                 # Documentação completa
├── requirements.txt          # Dependências gerais
├── data/                     # Dados do desafio
│   ├── train.csv
│   ├── test.csv
│   ├── sample_result.csv
│   └── result.csv            # RESPOSTA FINAL
├── notebooks/                # Análise exploratória
│   ├── 01_exploracao_inicial.ipynb
│   ├── 02_modelagem.ipynb
│   └── 03_analise_drift.ipynb
├── src/                      # Código fonte do modelo
│   └── models/
│       ├── train_final_model.py
│       └── random_forest_final.pkl
├── api/                      # API FastAPI (opcional)
│   ├── app.py
│   └── Dockerfile
└── streamlit/                # Aplicação web (BÔNUS)
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    ├── random_forest_final.pkl
    └── src/features/
        └── build_features.py
```
------------------------------------------------------------------------

## 🤖 Features do Modelo

O modelo utilizado foi um **Random Forest Classifier** treinado com
features derivadas do comportamento temporal da digitação.

| Feature           | Descrição                                                    |
|------------------|---------------------------------------------------------------|
| hold_mean        | Tempo médio em que uma tecla permanece pressionada            |
| hold_std         | Desvio padrão do tempo de pressão                             |
| keys_per_second  | Velocidade média de digitação                                 |
| flight_mean      | Intervalo médio entre soltar e pressionar                     |
| hold_cv          | Coeficiente de variação do hold time                          |

------------------------------------------------------------------------

## 🚀 Como Executar

### Gerar result.csv

python src/models/predict_test.py

------------------------------------------------------------------------

### Executar aplicação web

cd streamlit pip install -r requirements.txt streamlit run app.py

------------------------------------------------------------------------

### Executar com Docker

docker pull daniellsantanaa/keyboard-streamlit:latest docker run -p
8501:8501 daniellsantanaa/keyboard-streamlit:latest

------------------------------------------------------------------------

## 🌐 Aplicação Web

https://keyboard-app-180701311668.us-central1.run.app

Funcionalidades:

-   Simulador de digitação
-   Teste em tempo real
-   Upload de arquivos
-   Comparação entre perfis

------------------------------------------------------------------------

## 🐳 Docker Hub

https://hub.docker.com/r/daniellsantanaa/keyboard-streamlit


