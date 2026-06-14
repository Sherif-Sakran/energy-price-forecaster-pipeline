import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

# [NEW] Rate Limiting & LLM Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from google import genai
from google.genai import types

from app.features import FeatureEngineer

# 1. Dynamically Load the Pipeline Configuration
def load_config():
    config_path = Path(__file__).parent / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

CONFIG = load_config()

# [NEW] 2. Security Setup
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    # Compares the incoming key to a secret stored in your HF Space
    expected_key = os.getenv("CLIENT_ACCESS_TOKEN")
    if not expected_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header"
        )
    return api_key

# [NEW] 3. Rate Limiter Setup (Tracks requests by IP address)
limiter = Limiter(key_func=get_remote_address)

# 4. Request Schemas 
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
    price_lag_24h: float
    price_lag_48h: float
    price_lag_168h: float

# [NEW] LLM Request Schema (Strict Inputs Only)
class PostMortemRequest(BaseModel):
    model_name: str = Field(..., description="The evaluated model paradigm")
    target_date: str = Field(..., description="YYYY-MM-DD")
    rank_idx: str = Field(..., description="Rank index of the anomaly")
    aggregate_stats: dict = Field(..., description="Daily error metrics")
    hourly_timeline: list = Field(..., description="24-hour feature and error grid")

# [NEW] LLM Output Validation Schema (Matches your llm_agent.py)
class PostMortemReport(BaseModel):
    anomaly_date: str
    max_hourly_error_mwh: float
    primary_driver_failure: str
    narrative_explanation: str
    risk_mitigation_action: str

# 5. Global State
ml_models = {}
feature_engineer = FeatureEngineer(CONFIG)

# [NEW] Initialize Gemini Client
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Downloading model weights from Hugging Face Hub...")
    repo_id = os.getenv("HF_MODEL_REPO", "Sherif-Sakran/energy-price-forecaster")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id, 
            filename="paradigm_b_model.joblib",
            token=os.getenv("HF_TOKEN")
        )
        ml_models["model_b"] = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Startup warning/error: {e}")
    yield
    ml_models.clear()

app = FastAPI(title="Energy Price Forecaster API", lifespan=lifespan)

# [NEW] Attach the rate limiter to the FastAPI app instance
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def root():
    return {"message": "Welcome to the Energy Price Forecaster API. Use /predict for predictions and /generate-post-mortem for LLM analysis."}

@app.post("/predict")
def predict(request: PredictionRequest):
    model = ml_models.get("model_b")
    if not model:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    
    try:
        req_dict = request.model_dump(by_alias=True)
        timestamp = pd.to_datetime(req_dict.pop("timestamp"))
        
        df_raw = pd.DataFrame([req_dict], index=[timestamp])
        df_raw.index.name = 'date'
        
        target_col = CONFIG["features"]["target_col"]
        df_raw[target_col] = np.nan 
        
        df_base = feature_engineer.create_base_features(df_raw)
        X_processed, _, _ = feature_engineer.apply_categorical_and_cyclical_encodings(df_base)
        X_processed.columns = X_processed.columns.str.replace(' ', '_')
        
        missing_cols = [col for col in model.feature_name_ if col not in X_processed.columns]
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
            
        X_aligned = X_processed[model.feature_name_]
        prediction = model.predict(X_aligned)
        
        return {
            "timestamp": timestamp.isoformat(),
            "prediction_eur_mwh": float(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# [NEW] Protected LLM Endpoint
@app.post("/generate-post-mortem")
@limiter.limit("2/day") # Rate limit set here
def generate_post_mortem(
    request: Request, # Required by slowapi to track the IP
    payload: PostMortemRequest, 
    api_key: str = Security(get_api_key) # Authentication enforcement
):
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client is not configured.")

    # 1. Build the prompt using the validated payload
    import json
    system_prompt = f"""
    You are an advanced quantitative grid analytics engine evaluating model error post-mortems for a world-class company for energy trading.
    [ARCHITECTURAL METADATA]:
    - Model Paradigm Variant: {payload.model_name}
    - Evaluated Target Date: {payload.target_date}
    - Performance Severity Rank Index: {payload.rank_idx} 
    [TABULAR AGGREGATE SUMMARY STATE]:
    {json.dumps(payload.aggregate_stats, indent=2)}
    [24-HOUR HOURLY TIMELINE METRICS & FEATURES GRID]:
    {json.dumps(payload.hourly_timeline, indent=2)}
    [DIAGNOSTIC MANDATE]:
    1. Scan the 24-hour timeline to identify the exact hour clusters where the absolute error spiked.
    2. Cross-examine feature loads at those specific hours.
    3. Determine if the model missed a deep physical oversupply block.
    4. Formulate the response matching the strict output validation format parameters provided.
    """

    # 2. Execute the LLM Call
    try:
        response = gemini_client.models.generate_content(
            model=CONFIG["llm_agent"]["model"],
            contents=system_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PostMortemReport,
                temperature=CONFIG["llm_agent"]["temperature"],
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation Failed: {str(e)}")