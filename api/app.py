"""
API FastAPI para servir o modelo de classificação de digitação.
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn
import logging
import os
from typing import List, Dict, Any, Optional
import io
import json

# Importar funções de feature engineering
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.build_features import extract_features_from_json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar app
app = FastAPI(
    title="Keyboard Typing Classifier API",
    description="API para classificar digitações como humano ou bot",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja para origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos dos modelos
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_final.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'feature_names.pkl')

# Carregar modelo na inicialização
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    logger.info(f"✅ Modelo carregado: {MODEL_PATH}")
    logger.info(f"✅ Features esperadas: {len(feature_names)}")
except Exception as e:
    logger.error(f"❌ Erro ao carregar modelo: {e}")
    model = None
    feature_names = []

# Modelos Pydantic para request/response
class KeyEvent(BaseModel):
    code: int
    tick: int

class KeyboardData(BaseModel):
    keydown: List[KeyEvent]
    keyup: List[KeyEvent]

class InputData(BaseModel):
    keyboard: KeyboardData
    length: Optional[int] = None

class BatchPredictionRequest(BaseModel):
    inputs: List[Dict[str, Any]]

class SinglePredictionResponse(BaseModel):
    prediction: float
    probability_class_1: float
    probability_class_0: float
    is_bot: bool

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    statistics: Dict[str, float]

@app.get("/")
async def root():
    return {
        "message": "🔍 Keyboard Typing Classifier API",
        "model": "Random Forest",
        "version": "1.0.0",
        "status": "online",
        "endpoints": [
            "/health",
            "/predict",
            "/predict_batch",
            "/predict_file",
            "/feature_importance"
        ]
    }

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "RandomForest",
        "n_features": len(feature_names)
    }

@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(data: InputData):
    """
    Predição para uma única digitação.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Converter para formato esperado
        input_dict = {
            "keyboard": {
                "keydown": [e.dict() for e in data.keyboard.keydown],
                "keyup": [e.dict() for e in data.keyboard.keyup]
            },
            "length": data.length or len(data.keyboard.keydown)
        }
        
        # Extrair features
        df_features = extract_features_from_json(input_dict)
        
        # Garantir mesma ordem das features
        for feat in feature_names:
            if feat not in df_features.columns:
                df_features[feat] = 0
        
        X = df_features[feature_names]
        X = X.fillna(0)
        
        # Predizer
        prob = model.predict_proba(X)[0, 1]
        
        logger.info(f"Predição: {prob:.4f}")
        
        return SinglePredictionResponse(
            prediction=float(prob),
            probability_class_1=float(prob),
            probability_class_0=float(1 - prob),
            is_bot=prob > 0.5
        )
    
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predição em lote para múltiplas digitações.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Recebidas {len(request.inputs)} amostras")
        
        # Extrair features
        df_features = extract_features_from_json({"inputs": request.inputs})
        
        # Garantir features
        for feat in feature_names:
            if feat not in df_features.columns:
                df_features[feat] = 0
        
        X = df_features[feature_names].fillna(0)
        
        # Predizer
        predictions = model.predict_proba(X)[:, 1].tolist()
        
        # Estatísticas
        stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions))
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            statistics=stats
        )
    
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Upload de arquivo CSV ou JSON para predição.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            # Assumindo que o CSV tem coluna 'inputs'
            if 'inputs' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV deve ter coluna 'inputs'")
            
            predictions = []
            for inputs_str in df['inputs']:
                try:
                    input_dict = eval(inputs_str)  # cuidado: em produção use json.loads
                    df_feat = extract_features_from_json(input_dict)
                    for feat in feature_names:
                        if feat not in df_feat.columns:
                            df_feat[feat] = 0
                    X = df_feat[feature_names].fillna(0)
                    prob = model.predict_proba(X)[0, 1]
                    predictions.append(float(prob))
                except:
                    predictions.append(0.5)  # valor default em caso de erro
            
            result_df = pd.DataFrame({'target': predictions})
            result_csv = result_df.to_csv(index=False)
            
            return Response(
                content=result_csv,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={file.filename.replace('.csv', '_predictions.csv')}"}
            )
        
        elif file.filename.endswith('.json'):
            data = json.loads(content)
            return await predict_batch(BatchPredictionRequest(inputs=data if isinstance(data, list) else [data]))
        
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use CSV ou JSON.")
    
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/feature_importance")
async def get_feature_importance():
    """
    Retorna a importância das features do modelo.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    features = []
    for i in range(min(20, len(feature_names))):
        features.append({
            "feature": feature_names[indices[i]],
            "importance": float(importances[indices[i]])
        })
    
    return {"feature_importance": features}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)