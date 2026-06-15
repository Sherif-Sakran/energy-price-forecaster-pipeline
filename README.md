---
title: Energy Price Forecaster API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Energy Price Forecasting API

This repository contains the production inference API for the Day-Ahead European power market forecasting model. The system predicts electricity prices (EUR/MWh) using a LightGBM tabular model and includes a secured, LLM-powered anomaly diagnostics engine.

The API is built with **FastAPI**, containerised using **Docker**, and deployed continuously to **Hugging Face Spaces** via **GitHub Actions**.

---

## Core Features

* **Stateless Inference:** The API downloads the trained LightGBM model (`paradigm_b_model.joblib`) from the Hugging Face Hub directly into memory during server startup, preventing cold-start delays.
* **Synchronous Optimisation:** The prediction endpoint utilises FastAPI's background threadpool for CPU-bound model inference, preventing the main event loop from blocking under high concurrency.
* **Automated Feature Engineering:** Raw grid (megawatts) and weather (Celsius, m/s) parameters are ingested and transformed on the fly, calculating residual loads and cyclical time encodings dynamically.
* **Automated Post-Mortems:** A secured, rate-limited endpoint triggers a Gemini 2.5 Flash LLM agent to analyse prediction anomalies and generate structured diagnostic reports.
* **Continuous Deployment:** Every push to the `main` branch triggers a GitHub Action that seamlessly pushes the codebase to Hugging Face, rebuilding the Docker container automatically.

---

## Project Structure

```text
/
├── .github/
│   └── workflows/
│       └── deploy_to_hf.yml             # CI/CD GitHub Actions pipeline
├── app/      
│   ├── __init__.py      
│   ├── main.py                          # FastAPI application and endpoints
│   ├── features.py                      # Live feature engineering pipeline
│   └── pipeline_config.yaml             # Centralised configuration mapping
├── examples/   
│   ├── endpoint_llm_examples.txt        # Cases for calling the Gemini API
│   └── endpoint_predict_examples.txt    # A sample of prediction requests
├── Dockerfile                           # Hugging Face Spaces SDK specification
├── requirements.txt                     # API dependencies
└── README.md                            # HF config & project documentation
```

## API Endpoints
### 1. GET /
Checks the health of the endpoint.

### 2. 1. POST /predict
- Auth: None required.
- Input: JSON payload containing raw weather forecasts (temperature, wind speed) and grid metrics (solar, wind, load).
- Output: Predicted electricity price in EUR/MWh.

### 3. POST /generate-post-mortem
Triggers the quantitative grid analytics engine to evaluate model errors.
- Auth: Requires X-API-Key in the request header.
- Rate Limit: 2 requests per day per IP address.
- Input: JSON payload containing 24-hour evaluation metrics and feature grids.
- Output: A structured JSON diagnostic report (anomaly date, primary driver failure, risk mitigation actions).

## Local Development & Testing
### Option 1: Docker (Recommended for Production Parity)
1. Create your local environment file:
Create a .env file in the root directory:
```plaintext
HF_TOKEN=the_huggingface_access_token
HF_MODEL_REPO=Sherif-Sakran/energy-price-forecaster
CLIENT_ACCESS_TOKEN=generated_token_for_the_user
GEMINI_API_KEY=your_google_genai_key
```

2. Build the image locally:
```bash
docker build -t energy-price-forecaster .
```

3. Run the container (injecting the .env file):
```bash
docker run --env-file .env -p 7860:7860 energy-price-forecaster
```

### Option 2: Local without Docker
To run the API on your local machine without Docker:
1. Install Dependencies:
```bash
pip install -r requirements.txt
```
2. Set Environment Variables:
Create a .env file in the root directory with the following keys:
```bash
HF_TOKEN=the_huggingface_access_token
HF_MODEL_REPO=Sherif-Sakran/energy-price-forecaster
CLIENT_ACCESS_TOKEN=generated_token_for_the_user
GEMINI_API_KEY=your_google_genai_key
```
3. Start the Server: 
```bash
uvicorn app.main:app --reload --port 7860
```

### Dodumentation Access
For both methods, the interactive documentation and testing UI will be available at 
`http://localhost:7860/docs`
