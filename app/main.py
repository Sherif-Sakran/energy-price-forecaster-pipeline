import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

from app.features import FeatureEngineer

# 1. Dynamically Load the Pipeline Configuration
def load_config():
    config_path = Path(__file__).parent / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

CONFIG = load_config()

# 2. Define Request Schema 
class PredictionRequest(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 format, e.g., '2026-04-09T14:00:00+02:00'")
    
    # Future Forecast (Context)
    forecasted_load: float = Field(..., alias="Forecasted Load")
    solar: float = Field(..., alias="Solar")
    wind_offshore: float = Field(..., alias="Wind Offshore")
    wind_onshore: float = Field(..., alias="Wind Onshore")
    temperature_2m: float 
    wind_speed_120m: float
    is_holiday: int 
    
    # Price Anchor
    price_lag_24h: float
    price_lag_48h: float
    price_lag_168h: float

ml_models = {}
feature_engineer = FeatureEngineer(CONFIG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Downloading model weights from Hugging Face Hub...")
    repo_id = os.getenv("HF_MODEL_REPO", "Sherif-Sakran/energy-price-forecaster")
    
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="paradigm_b_model.joblib")
        ml_models["model_b"] = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Startup warning/error: {e}")
    yield
    ml_models.clear()

app = FastAPI(title="Eergy Price Forecaster API", lifespan=lifespan)

@app.post("/")
async def root():
    return {"message": "Welcome to the Energy Price Forecaster API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
async def predict(request: PredictionRequest):
    model = ml_models.get("model_b")
    if not model:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    
    try:
        req_dict = request.model_dump(by_alias=True)
        timestamp = pd.to_datetime(req_dict.pop("timestamp"))
        
        # Reconstruct DataFrame
        df_raw = pd.DataFrame([req_dict], index=[timestamp])
        df_raw.index.name = 'date'
        
        # Inject dummy target column to bypass target separation errors in features.py
        target_col = CONFIG["features"]["target_col"]
        df_raw[target_col] = np.nan 
        
        # Apply Feature Engineering
        df_base = feature_engineer.create_base_features(df_raw)
        X_processed, _, _ = feature_engineer.apply_categorical_and_cyclical_encodings(df_base)

        # LightGBM silently replaced spaces with underscores during training.
        X_processed.columns = X_processed.columns.str.replace(' ', '_')

        # Feature Alignment check
        missing_cols = [col for col in model.feature_name_ if col not in X_processed.columns]
        if missing_cols:
            raise ValueError(f"Missing required features after processing: {missing_cols}")
            
        X_aligned = X_processed[model.feature_name_]
        
        # Predict
        prediction = model.predict(X_aligned)
        
        return {
            "timestamp": timestamp.isoformat(),
            "prediction_eur_mwh": float(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction pipeline failed: {str(e)}")  